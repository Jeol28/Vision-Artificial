"""Carga de los modelos transformer e inferencia compatible con ZeroGPU.

**ZeroGPU** (Hugging Face Spaces) asigna una GPU solo mientras se ejecuta una
función decorada con ``@spaces.GPU``. Sus reglas:

* los modelos se cargan **al arrancar** el proceso y se mueven a ``cuda`` ahí
  mismo (ZeroGPU los empaqueta y los lleva a la GPU en cada llamada);
* la inferencia va dentro de funciones ``@spaces.GPU``; lo que devuelven debe
  poder serializarse (aquí: diccionarios con NumPy, nunca tensores ni modelos);
* ``duration`` es la reserva de GPU por llamada y se descuenta de la cuota diaria
  del visitante antes de ejecutar: si es mayor que lo que le queda, la llamada
  falla. Por eso se piden reservas cortas (la inferencia real tarda milisegundos).

Fuera de un Space con ZeroGPU, ``spaces.GPU`` no hace nada y los modelos se
cargan la primera vez que se usan (GPU NVIDIA, GPU Apple o CPU). En los dos
casos cada modelo se carga **una sola vez** por proceso.
"""

from __future__ import annotations

import os
import threading

from volleyvision.core import pipeline as pl
from volleyvision.core import transformers_models as tm

try:  # en un Space, ``spaces`` debe importarse antes que torch (lo hace app.py)
    import spaces

    GPU = spaces.GPU
except ImportError:  # pragma: no cover - sin el paquete ``spaces`` el decorador no hace nada
    def GPU(funcion=None, **_):
        return funcion if callable(funcion) else (lambda f: f)

EN_ZEROGPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "t", "true")

_cache: dict[str, tm.ModeloCargado] = {}
_candado = threading.Lock()


def _obtener(clave: str, cargar) -> tm.ModeloCargado:
    with _candado:
        if clave not in _cache:
            _cache[clave] = cargar()
        return _cache[clave]


def detector(nombre: str) -> tm.ModeloCargado:
    """RT-DETR o DETR por su nombre en ``tm.MODELOS_DETECCION``."""
    id_hf = tm.MODELOS_DETECCION[nombre]

    def cargar():
        m = tm.cargar_detector(id_hf)
        if not EN_ZEROGPU:  # con ZeroGPU se calienta dentro de cada llamada en GPU
            tm.calentar(m)  # la primera inferencia es más lenta: no debe contar en la comparación
        return m

    return _obtener(id_hf, cargar)


def segformer() -> tm.ModeloCargado:
    return _obtener(tm.MODELO_SEGFORMER, tm.cargar_segformer)


def sam(nombre: str) -> tm.ModeloCargado:
    id_hf = tm.MODELOS_SAM[nombre]
    return _obtener(id_hf, lambda: tm.cargar_sam(id_hf))


def precargar() -> None:
    """Carga todos los modelos al arrancar (obligatorio con ZeroGPU)."""
    for nombre in tm.MODELOS_DETECCION:
        detector(nombre)
    segformer()
    for nombre in tm.MODELOS_SAM:
        sam(nombre)


def info(m: tm.ModeloCargado) -> str:
    """Línea con el enlace, el tamaño y el dispositivo del modelo."""
    disp = "GPU ZeroGPU (cuda)" if EN_ZEROGPU else m.dispositivo
    return f"[{m.id_hf}]({m.enlace}) · {m.n_parametros / 1e6:.1f} M parámetros · {disp}"


def _calentar_en_gpu(det: tm.ModeloCargado) -> None:
    # Con ZeroGPU cada llamada puede caer en una GPU distinta: se calienta antes de medir.
    if EN_ZEROGPU:
        tm.calentar(det)


# --------------------------------------------------------------------------
# Inferencia (estas funciones usan la GPU de ZeroGPU cuando existe)
# --------------------------------------------------------------------------
@GPU(duration=15)
def detectar(img, nombre: str, umbral: float) -> dict:
    det = detector(nombre)
    _calentar_en_gpu(det)
    return tm.detectar_objetos(img, det, umbral)


@GPU(duration=15)
def segmentar_semantica(img) -> dict:
    return tm.segmentar_semantica(img, segformer())


@GPU(duration=20)
def segmentar_con_sam(img, nombre: str, punto=None, caja=None) -> dict:
    return tm.segmentar_con_sam(img, sam(nombre), punto=punto, caja=caja)


@GPU(duration=30)
def comparar(img, nombres: list[str], umbral: float, verdad=None) -> dict:
    detectores = {n: detector(n) for n in nombres}
    for d in detectores.values():
        _calentar_en_gpu(d)
    return tm.comparar_metodos(img, detectores, umbral, verdad)


@GPU(duration=30)
def pipeline_con_modelos(img, cfg: pl.ConfigPipeline, nombre_det: str | None, usar_seg: bool):
    det = detector(nombre_det) if nombre_det else None
    if det is not None:
        _calentar_en_gpu(det)
    return pl.ejecutar_pipeline(img, cfg, det, segformer() if usar_seg else None)
