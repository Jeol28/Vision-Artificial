"""Pruebas de la integración con transformers y del pipeline.

Las pruebas marcadas ``slow`` descargan modelos de Hugging Face; se ejecutan con
``pytest -m slow``. Las demás usan solo la vía clásica.
"""

import numpy as np
import pytest

from volleyvision.core import pipeline as pl
from volleyvision.core import transformers_models as tm


def test_dispositivo_valido():
    assert tm.elegir_dispositivo() in {"cpu", "cuda", "mps"}


def test_detector_clasico_mismo_formato(escena):
    res = tm.detectar_clasico(escena.imagen)
    resumen = tm.resumen_detecciones(res)
    assert resumen["balones"] == 1
    assert resumen["personas"] >= 7  # oclusiones de la cámara trasera
    assert all(set(d) == {"etiqueta", "puntaje", "caja"} for d in res["detecciones"])


def test_pipeline_clasico_completo(escena):
    cfg = pl.ConfigPipeline(esquinas=escena.esquinas)
    pasos = pl.ejecutar_pipeline(escena.imagen, cfg)
    assert [p.nombre[0] for p in pasos] == list("1234567")
    mapa = pasos[-1]
    assert "Mapa de posiciones" in mapa.imagenes
    propios = mapa.datos["posiciones_m"]
    assert len(propios) >= 4  # solo tu equipo (6 en la escena; el clásico une a algunos de la red)
    assert all(p["y (m)"] >= 8.5 for p in propios)  # todos en tu media cancha
    datos = pasos[5].datos
    assert datos["Jugadores de tu equipo"] == len(propios)
    assert datos["Descartados (rival o fuera de la cancha)"] >= 1  # el rival no se analiza


def test_pipeline_con_calibracion(escena, vistas_tablero):
    from volleyvision.core import calibracion as cal
    from volleyvision.utils import sinteticos

    res = cal.calibrar_camara(vistas_tablero, sinteticos.PATRON, sinteticos.TAM_CUADRO_MM)
    pasos = pl.ejecutar_pipeline(escena.imagen, pl.ConfigPipeline(calibracion=res, esquinas=escena.esquinas))
    assert pasos[0].imagenes["Corregida"].shape == escena.imagen.shape
    assert "Mapa de posiciones" in pasos[-1].imagenes


def test_pipeline_sin_esquinas_omite_mapa(escena):
    pasos = pl.ejecutar_pipeline(escena.imagen)
    assert pasos[-1].imagenes == {}
    assert "esquinas" in pasos[-1].nota


@pytest.mark.slow
def test_detectores_transformer_reales():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    for id_hf in tm.MODELOS_DETECCION.values():
        det = tm.cargar_detector(id_hf, "cpu")
        res = tm.detectar_objetos(img, det, 0.5)
        assert res["tiempo_ms"] > 0
        assert isinstance(res["detecciones"], list)


@pytest.mark.slow
def test_segformer_y_sam(escena):
    seg = tm.cargar_segformer(dispositivo="cpu")
    sem = tm.segmentar_semantica(escena.imagen, seg)
    assert sem["etiquetas"].shape == escena.imagen.shape[:2]
    sam = tm.cargar_sam(dispositivo="cpu")
    r = tm.segmentar_con_sam(escena.imagen, sam, punto=escena.balon["centro"])
    cx, cy = escena.balon["centro"]
    assert r["mascara"][cy, cx] == 255
    assert (r["mascara"] > 0).sum() < 5000  # segmenta el balón, no la escena
