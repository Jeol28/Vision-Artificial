"""Fotos reales de ejemplo (data/samples/reales/): esquinas guardadas y análisis del equipo propio."""

import cv2
import numpy as np
import pytest

from volleyvision.core import calibracion as cal
from volleyvision.core import pipeline as pl
from volleyvision.ui import componentes as ui
from volleyvision.utils import io

REALES = sorted(p.name for p in io.DIR_REALES.glob("*.jpg"))


def test_hay_fotos_reales_y_son_la_muestra_por_defecto():
    assert len(REALES) >= 6
    assert ui.MUESTRA_POR_DEFECTO in REALES
    assert list(io.listar_muestras())[: len(REALES)] == REALES  # las reales van primero


@pytest.mark.parametrize("nombre", REALES)
def test_cada_foto_trae_las_esquinas_de_su_media_cancha(nombre):
    img, _ = ui.resolver_imagen(*ui.valores_por_defecto(nombre))
    esq = ui.esquinas_ejemplo(nombre, img)
    assert esq is not None and esq.shape == (4, 2)
    h, w = img.shape[:2]
    assert (esq[:, 0] >= 0).all() and (esq[:, 0] <= w).all() and (esq[:, 1] >= 0).all() and (esq[:, 1] <= h).all()
    assert cv2.isContourConvex(esq.astype(np.float32).reshape(-1, 1, 2))
    # la red (lado superior) queda más lejos de la cámara (más arriba) que la línea de fondo propia
    assert esq[:2, 1].mean() < esq[2:, 1].mean()
    # la homografía lleva las esquinas a la media cancha de 9 × 9 m
    m = cal.imagen_a_cancha(esq, cal.homografia_cancha(esq))
    assert np.allclose(m, cal.ESQUINAS_MEDIA_CANCHA_M, atol=1e-3)


def test_pipeline_clasico_en_foto_real():
    img, nombre = ui.resolver_imagen(*ui.valores_por_defecto())
    pasos = pl.ejecutar_pipeline(img, pl.ConfigPipeline(esquinas=ui.esquinas_ejemplo(nombre, img)))
    datos = pasos[5].datos
    assert datos["Jugadores de tu equipo"] >= 3 and datos["Descartados (rival o fuera de la cancha)"] >= 5
    for p in pasos[6].datos["posiciones_m"]:  # nadie del lado rival ni de los costados
        assert p["y (m)"] >= 8.8 and -0.5 <= p["x (m)"] <= 9.5
