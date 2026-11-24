r"""Modelos transformer de Hugging Face (solo inferencia, pesos públicos).

No se entrena ni se etiqueta nada: se cargan pesos preentrenados y se usan tal cual.

Detección: DETR y RT-DETR
-------------------------
**DETR** (*DEtection TRansformer*, Carion et al., 2020) trata la detección como
predicción de un **conjunto**: una CNN (ResNet-50) extrae características, un
codificador transformer las relaciona globalmente con auto-atención

.. math::

    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V

y un decodificador con :math:`N = 100` *object queries* produce directamente
N pares (clase, caja). En el entrenamiento, el emparejamiento húngaro asigna
cada predicción a un objeto real, así que no hace falta NMS ni anclas.

**RT-DETR** (*Real-Time DETR*, Zhao et al., 2023) conserva la idea pero usa un
codificador híbrido eficiente (atención intra-escala + fusión CNN entre
escalas) y selecciona las queries iniciales por su puntaje IoU. Es varias veces
más rápido que DETR con mejor precisión. Ambos están entrenados en COCO, que
incluye las clases ``person`` y ``sports ball``.

Segmentación semántica: SegFormer
---------------------------------
**SegFormer** (Xie et al., 2021) usa un codificador transformer jerárquico
(*Mix Transformer*) sin codificación posicional y un decodificador MLP
ligero. Asigna una clase a cada píxel. El modelo B0 entrenado en ADE20K
(150 clases) distingue ``person``, ``floor``, ``wall``, etc.

Segmentación por *prompt*: SAM / SlimSAM
----------------------------------------
**SAM** (*Segment Anything*, Kirillov et al., 2023) tiene un codificador de
imagen ViT pesado (se ejecuta una vez), un codificador de *prompts* (puntos,
cajas) y un decodificador de máscaras ligero que propone 3 máscaras con su IoU
estimado. **SlimSAM** es una versión podada y destilada (~10 % de los
parámetros), práctica en CPU.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

MODELOS_DETECCION = {
    "RT-DETR (PekingU/rtdetr_r50vd)": "PekingU/rtdetr_r50vd",
    "DETR (facebook/detr-resnet-50)": "facebook/detr-resnet-50",
}
MODELO_SEGFORMER = "nvidia/segformer-b0-finetuned-ade-512-512"
MODELOS_SAM = {
    "SlimSAM (nielsr/slimsam-50-uniform)": "nielsr/slimsam-50-uniform",
    "SAM ViT-B (facebook/sam-vit-base)": "facebook/sam-vit-base",
}
CLASES_VOLEIBOL = ("person", "sports ball")
ENLACE_HF = "https://huggingface.co/{}"


def elegir_dispositivo() -> str:
    """GPU NVIDIA (``cuda``) → GPU Apple (``mps``) → ``cpu``.

    Se puede forzar con la variable de entorno ``VOLLEYVISION_DEVICE``.
    """
    import torch

    forzado = os.environ.get("VOLLEYVISION_DEVICE")
    if forzado:
        return forzado
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sincronizar(dispositivo: str) -> None:
    """Espera a que termine el trabajo en la GPU para medir tiempos reales."""
    import torch

    if dispositivo == "cuda":
        torch.cuda.synchronize()
    elif dispositivo == "mps":
        torch.mps.synchronize()


@dataclass
class ModeloCargado:
    """Procesador + modelo de Hugging Face listos para inferencia."""

    id_hf: str
    procesador: object
    modelo: object
    dispositivo: str
    tiempo_carga_s: float
    extra: dict = field(default_factory=dict)

    @property
    def enlace(self) -> str:
        return ENLACE_HF.format(self.id_hf)

    @property
    def n_parametros(self) -> int:
        return sum(p.numel() for p in self.modelo.parameters())


def _cargar(id_hf: str, clase_modelo, clase_procesador, dispositivo: str | None) -> ModeloCargado:
    dispositivo = dispositivo or elegir_dispositivo()
    t0 = time.perf_counter()
    procesador = clase_procesador.from_pretrained(id_hf)
    modelo = clase_modelo.from_pretrained(id_hf).eval().to(dispositivo)
    return ModeloCargado(id_hf, procesador, modelo, dispositivo, time.perf_counter() - t0)


def cargar_detector(id_hf: str = "PekingU/rtdetr_r50vd", dispositivo: str | None = None) -> ModeloCargado:
    """Carga DETR o RT-DETR (los pesos se descargan la primera vez a la caché de HF)."""
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    return _cargar(id_hf, AutoModelForObjectDetection, AutoImageProcessor, dispositivo)


def cargar_segformer(id_hf: str = MODELO_SEGFORMER, dispositivo: str | None = None) -> ModeloCargado:
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    return _cargar(id_hf, SegformerForSemanticSegmentation, AutoImageProcessor, dispositivo)


def cargar_sam(id_hf: str = "nielsr/slimsam-50-uniform", dispositivo: str | None = None) -> ModeloCargado:
    from transformers import SamModel, SamProcessor

    return _cargar(id_hf, SamModel, SamProcessor, dispositivo)


def _a_dispositivo(entradas, dispositivo: str):
    """Mueve las entradas al dispositivo; los tensores float64 pasan a float32 (MPS no admite float64)."""
    import torch

    for k, v in list(entradas.items()):
        if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
            entradas[k] = v.float()
    return entradas.to(dispositivo)


def _a_pil(img_bgr: np.ndarray):
    from PIL import Image

    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# --------------------------------------------------------------------------
# Detección
# --------------------------------------------------------------------------
def detectar_objetos(img: np.ndarray, det: ModeloCargado, umbral: float = 0.5,
                     clases: tuple[str, ...] | None = CLASES_VOLEIBOL) -> dict:
    """Detecta objetos con DETR / RT-DETR.

    Returns:
        ``detecciones`` (lista de ``{"etiqueta", "puntaje", "caja"}``), ``tiempo_ms``
        (solo la red, sin pre/posproceso) y ``tiempo_total_ms``.
    """
    import torch

    t0 = time.perf_counter()
    pil = _a_pil(img)
    entradas = _a_dispositivo(det.procesador(images=pil, return_tensors="pt"), det.dispositivo)
    _sincronizar(det.dispositivo)
    t1 = time.perf_counter()
    with torch.no_grad():
        salidas = det.modelo(**entradas)
    _sincronizar(det.dispositivo)
    t2 = time.perf_counter()
    res = det.procesador.post_process_object_detection(salidas, threshold=float(umbral),
                                                       target_sizes=[(img.shape[0], img.shape[1])])[0]
    id2label = det.modelo.config.id2label
    detecciones = []
    for p, l, c in zip(res["scores"].cpu().tolist(), res["labels"].cpu().tolist(), res["boxes"].cpu().tolist()):
        etiqueta = id2label[int(l)]
        if clases is None or etiqueta in clases:
            detecciones.append({"etiqueta": etiqueta, "puntaje": float(p), "caja": tuple(float(v) for v in c)})
    detecciones.sort(key=lambda d: d["puntaje"], reverse=True)
    return {"detecciones": detecciones, "tiempo_ms": (t2 - t1) * 1e3,
            "tiempo_total_ms": (time.perf_counter() - t0) * 1e3}


def calentar(det: ModeloCargado, tam=(480, 640)) -> None:
    """Una inferencia de prueba: la primera llamada incluye inicializaciones que no
    deben contarse al comparar tiempos."""
    detectar_objetos(np.zeros((*tam, 3), np.uint8), det, umbral=0.99)


# --------------------------------------------------------------------------
# Segmentación semántica (SegFormer)
# --------------------------------------------------------------------------
def segmentar_semantica(img: np.ndarray, seg: ModeloCargado) -> dict:
    """Segmentación semántica por píxel.

    Returns:
        ``etiquetas`` (H×W con el id de clase), ``id2label``, ``clases_presentes``
        (lista de ``(nombre, fracción del área)``) y ``tiempo_ms``.
    """
    import torch

    entradas = _a_dispositivo(seg.procesador(images=_a_pil(img), return_tensors="pt"), seg.dispositivo)
    _sincronizar(seg.dispositivo)
    t0 = time.perf_counter()
    with torch.no_grad():
        salidas = seg.modelo(**entradas)
    _sincronizar(seg.dispositivo)
    dt = (time.perf_counter() - t0) * 1e3
    mapa = seg.procesador.post_process_semantic_segmentation(salidas, target_sizes=[img.shape[:2]])[0]
    etiquetas = mapa.cpu().numpy().astype(np.int32)
    id2label = seg.modelo.config.id2label
    ids, cuentas = np.unique(etiquetas, return_counts=True)
    presentes = sorted(((id2label[int(i)], c / etiquetas.size) for i, c in zip(ids, cuentas)),
                       key=lambda x: -x[1])
    return {"etiquetas": etiquetas, "id2label": id2label, "clases_presentes": presentes, "tiempo_ms": dt}


def mascara_de_clase(etiquetas: np.ndarray, id2label: dict, nombre: str = "person") -> np.ndarray:
    """Máscara binaria (0/255) de una clase de la segmentación semántica."""
    ids = [int(i) for i, n in id2label.items() if n == nombre]
    return np.isin(etiquetas, ids).astype(np.uint8) * 255


# --------------------------------------------------------------------------
# SAM / SlimSAM
# --------------------------------------------------------------------------
def segmentar_con_sam(img: np.ndarray, sam: ModeloCargado, punto=None, caja=None) -> dict:
    """Segmenta el objeto indicado por un punto ``(x, y)`` y/o una caja ``(x1, y1, x2, y2)``.

    SAM propone 3 máscaras (parte, objeto, todo) con su IoU estimado; se elige la de mayor IoU.

    Returns:
        ``mascara`` (uint8 0/255), ``puntaje`` (IoU estimado), ``mascaras`` (las 3),
        ``puntajes`` y ``tiempo_ms``.
    """
    import torch

    if punto is None and caja is None:
        raise ValueError("SAM necesita un punto o una caja como prompt.")
    kwargs = {}
    if punto is not None:
        kwargs["input_points"] = [[[float(punto[0]), float(punto[1])]]]
    if caja is not None:
        kwargs["input_boxes"] = [[[float(v) for v in caja]]]
    entradas = _a_dispositivo(sam.procesador(images=_a_pil(img), return_tensors="pt", **kwargs), sam.dispositivo)
    _sincronizar(sam.dispositivo)
    t0 = time.perf_counter()
    with torch.no_grad():
        salidas = sam.modelo(**entradas)
    _sincronizar(sam.dispositivo)
    dt = (time.perf_counter() - t0) * 1e3
    mascaras = sam.procesador.image_processor.post_process_masks(
        salidas.pred_masks.cpu(), entradas["original_sizes"].cpu(), entradas["reshaped_input_sizes"].cpu())[0]
    mascaras = mascaras[0].numpy()               # (3, H, W) booleano
    puntajes = salidas.iou_scores[0, 0].cpu().numpy()
    mejor = int(np.argmax(puntajes))
    return {"mascara": mascaras[mejor].astype(np.uint8) * 255, "puntaje": float(puntajes[mejor]),
            "mascaras": [(m.astype(np.uint8) * 255) for m in mascaras], "puntajes": puntajes.tolist(),
            "tiempo_ms": dt}


# --------------------------------------------------------------------------
# Comparación clásico vs. transformer
# --------------------------------------------------------------------------
def detectar_clasico(img: np.ndarray, **params) -> dict:
    """Detección sin aprendizaje (mismo formato que :func:`detectar_objetos`):
    jugadores con k-means + morfología y balón con color + forma."""
    from volleyvision.core import caracterizacion as ca
    from volleyvision.core import segmentacion as sg

    t0 = time.perf_counter()
    balones = [c for c in ca.detectar_balon_clasico(img) if c["es_balon"]]
    excluir = np.zeros(img.shape[:2], np.uint8)
    for b in balones:
        cv2.drawContours(excluir, [b["contorno"]], -1, 255, -1)
    excluir = cv2.dilate(excluir, np.ones((7, 7), np.uint8))
    jug = sg.segmentar_jugadores_clasico(img, mascara_excluir=excluir, **params)
    dt = (time.perf_counter() - t0) * 1e3
    dets = [{"etiqueta": "person", "puntaje": 1.0, "caja": tuple(float(v) for v in r["caja"])}
            for r in jug["regiones"]]
    dets += [{"etiqueta": "sports ball", "puntaje": float(b["puntaje"]), "caja": tuple(float(v) for v in b["caja"])}
             for b in balones]
    return {"detecciones": dets, "tiempo_ms": dt, "tiempo_total_ms": dt}


def resumen_detecciones(res: dict) -> dict:
    """Cuenta personas y balones de un resultado de detección."""
    dets = res["detecciones"]
    return {"personas": sum(d["etiqueta"] == "person" for d in dets),
            "balones": sum(d["etiqueta"] == "sports ball" for d in dets),
            "tiempo_ms": res["tiempo_ms"]}


def cajas_para_dibujar(dets: list[dict], correctas: list[bool] | None = None) -> list[dict]:
    """Convierte detecciones al formato de :func:`utils.visualizacion.dibujar_cajas`.

    Sin verdad conocida: personas en naranja y balón en amarillo. Con verdad
    (``correctas``): verde = acierto, rojo = falso positivo.
    """
    from volleyvision.utils.visualizacion import NARANJA, ROJO, VERDE

    salida = []
    for i, d in enumerate(dets):
        if correctas is not None:
            color = VERDE if correctas[i] else ROJO
        else:
            color = NARANJA if d["etiqueta"] == "person" else (0, 230, 255)
        nombre = "persona" if d["etiqueta"] == "person" else "balón"
        salida.append({"caja": d["caja"], "etiqueta": f"{nombre} {d['puntaje']:.2f}", "color": color})
    return salida


def comparar_metodos(img: np.ndarray, detectores: dict[str, ModeloCargado], umbral: float = 0.5,
                     verdad: list | None = None) -> dict[str, dict]:
    """Ejecuta la detección clásica y cada detector transformer sobre la misma imagen.

    Args:
        detectores: nombre → modelo cargado.
        verdad: cajas reales de personas (si se conocen, p. ej. en la escena sintética)
            para calcular aciertos.

    Returns:
        nombre del método → ``{"resultado", "resumen", "evaluacion"}``.
    """
    from volleyvision.core import segmentacion as sg

    resultados = {"Clásico (k-means + morfología)": detectar_clasico(img)}
    for nombre, det in detectores.items():
        resultados[nombre] = detectar_objetos(img, det, umbral)
    salida = {}
    for nombre, res in resultados.items():
        ev = None
        if verdad is not None:
            pred = [d["caja"] for d in res["detecciones"] if d["etiqueta"] == "person"]
            ev = sg.emparejar_cajas(pred, verdad, 0.3)
        salida[nombre] = {"resultado": res, "resumen": resumen_detecciones(res), "evaluacion": ev}
    return salida
