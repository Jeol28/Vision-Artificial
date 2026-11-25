"""Pruebas de la UI (Gradio): la app se construye y cada callback corre sin excepciones.

Los callbacks se llaman directamente con los valores por defecto del selector
de imagen (una foto real de ejemplo con sus esquinas), sin levantar un servidor.
"""

import gradio as gr
import numpy as np
import pytest

from volleyvision.ui import componentes as ui
from volleyvision.ui.app import construir_app
from volleyvision.ui.paginas import (p1_calibracion, p2_filtros, p3_histogramas, p4_umbralizacion, p5_morfologia,
                                     p6_segmentacion, p7_caracterizacion, p8_transformers, p9_pipeline)

SEL = ui.valores_por_defecto()


def _valor(salida):
    """Valor de una salida que puede venir envuelta en un componente (``gr.Image(valor, label=...)``)."""
    return salida.value if isinstance(salida, gr.components.Component) else salida


def test_app_se_construye():
    demo = construir_app()
    assert isinstance(demo, gr.Blocks)


def test_resolver_imagen_por_defecto():
    img, nombre = ui.resolver_imagen(*SEL)
    assert img.ndim == 3 and nombre == ui.MUESTRA_POR_DEFECTO


def test_resolver_frame_de_video(tmp_path):
    from volleyvision.utils import sinteticos
    ruta = tmp_path / "mi_partido.mp4"  # video subido por el usuario (aquí, uno generado para la prueba)
    sinteticos.generar_video_sintetico(ruta)
    img, nombre = ui.resolver_imagen(ui.FUENTES[2], None, None, ui.SUBIR_VIDEO, str(ruta), 10)
    assert img.ndim == 3 and nombre == "mi_partido.mp4"


def test_subir_sin_archivo_da_error():
    with pytest.raises(gr.Error):
        ui.resolver_imagen(ui.FUENTES[1], None, None, None, None, 0)


def test_calibracion_y_homografia():
    estado = ui.estado_nuevo()
    *_, estado = p1_calibracion.calibrar(p1_calibracion.FUENTES_TABLERO[0], None, 9, 6, 25.0, estado)
    assert estado["calibracion"].error_rms < 0.5
    antes, despues = p1_calibracion.corregir(estado["calibracion"].indices_validos[0], 0.0, estado)
    assert antes.shape == despues.shape

    vista, _ = p1_calibracion.homografia(*SEL, 30, 2.0, estado)
    assert vista is None  # aún no hay esquinas
    img, nombre = ui.resolver_imagen(*SEL)
    estado["esquinas"][nombre] = ui.esquinas_ejemplo(nombre, img).tolist()
    vista, texto = p1_calibracion.homografia(*SEL, 30, 2.0, estado)
    assert vista.shape == (22 * 30, 13 * 30, 3) and "DLT" in texto


@pytest.mark.parametrize("llamada", [
    lambda: p2_filtros.suavizado(*SEL, "Bilateral", 9, "Sal y pimienta", 0, 75, 75),
    lambda: p2_filtros.bordes(*SEL, 3, 50, 150, 5),
    lambda: p2_filtros.convolucion(*SEL, "1 0 -1\n2 0 -2\n1 0 -1", "absoluto", True),
    lambda: p2_filtros.lineas(*SEL, 9, 50, 150, 80, 60, 10),
    lambda: p3_histogramas.histograma(*SEL, "HSV"),
    lambda: p3_histogramas.ecualizacion(*SEL, 2.0, 8),
    lambda: p3_histogramas.uniformes(*SEL, p3_histogramas.FUENTES_JUGADORES[0], "Bhattacharyya", None),
    lambda: p4_umbralizacion.umbral_global(*SEL, 127, False),
    lambda: p4_umbralizacion.otsu(*SEL),
    lambda: p4_umbralizacion.adaptativo(*SEL, "gaussiano", 31, 5, True),
    lambda: p4_umbralizacion.hsv(*SEL, 20, 35, 100, 255, 100, 255, False),
    lambda: p5_morfologia.morfologia(*SEL, p5_morfologia.MASCARAS[1], 1.0, "Apertura", "Elipse", 5, 1, 3, 7, 40),
    lambda: p6_segmentacion.kmeans(*SEL, 6, "Lab"),
    lambda: p6_segmentacion.contornos(*SEL, 150),
    lambda: p6_segmentacion.watershed(*SEL, p6_segmentacion.DEMOS_WS[1], 0.6),
    lambda: p6_segmentacion.jugadores(*SEL, 6, 9, False),
    lambda: p6_segmentacion.cajas_clasicas(*SEL),
    lambda: p7_caracterizacion.opciones_regiones(*SEL, p7_caracterizacion.REGIONES[0]),
    lambda: p7_caracterizacion.descriptores(*SEL, p7_caracterizacion.REGIONES[0], 0),
    lambda: p7_caracterizacion.es_balon(*SEL, 0.78, 0.45, 0.85, 0.4),
], ids=lambda f: "callback")
def test_callbacks_clasicos(llamada):
    salida = llamada()
    assert salida is not None


def test_otsu_manual_igual_a_opencv_en_ui():
    tarjetas, *_ = p4_umbralizacion.otsu(*SEL)
    assert "Diferencia</div><div class='val'>0<" in tarjetas


def test_watershed_separa_los_balones_de_la_demo():
    _, texto = p6_segmentacion.watershed(*SEL, p6_segmentacion.DEMOS_WS[1], 0.6)
    assert "Contornos encontrarían **1**" in texto and "watershed encuentra **2**" in texto


def test_jugadores_clasicos():
    texto, galeria = p6_segmentacion.jugadores(*SEL, 6, 9, False)
    assert "regiones con forma de persona" in texto and len(galeria) == 3


def test_proyeccion_a_la_cancha():
    estado = ui.estado_nuevo()
    *_, nota = p7_caracterizacion.proyeccion(*SEL, estado)
    assert "Marca" in nota
    img, nombre = ui.resolver_imagen(*SEL)
    estado["esquinas"][nombre] = ui.esquinas_ejemplo(nombre, img).tolist()
    apoyo, mapa, tabla, nota = p7_caracterizacion.proyeccion(*SEL, estado)
    assert nota == "" and (tabla["¿tu equipo?"] != "no").sum() >= 4 and (tabla["¿tu equipo?"] == "no").sum() >= 1
    assert isinstance(apoyo, np.ndarray) and _valor(mapa) is not None


def test_pipeline_clasico_desde_ui():
    estado = ui.estado_nuevo()
    tarjetas, fig, pasos = p9_pipeline.ejecutar(*SEL, False, True, 2.0, 6, 9, False, None, 0.5, False, estado)
    assert "Tiempo total" in tarjetas and len(pasos) == 7
    assert pasos[-1].imagenes  # con las esquinas de ejemplo hay mapa cenital
    assert "✅" in p9_pipeline.estado_esquinas(*SEL, True, estado)


@pytest.mark.slow
def test_pagina_transformers():
    nombre = next(iter(p8_transformers.tm.MODELOS_DETECCION))
    dibujo, tarjetas, info, tabla = p8_transformers.deteccion(*SEL, nombre, 0.5)
    assert "Personas" in tarjetas and "huggingface.co" in info
    gal, tabla, fig, nota = p8_transformers.comparacion(*SEL, 0.5, list(p8_transformers.tm.MODELOS_DETECCION))
    assert len(gal) == 3 and "tiempo (ms)" in tabla.columns


def test_analizar_modo_guiado():
    from volleyvision.ui.paginas import analizar
    estado = ui.estado_nuevo()
    assert "✅" in analizar.estado_esquinas(*SEL, estado)  # la escena de ejemplo trae sus esquinas
    anotada, mapa, resumen, pasos = analizar.analizar(*SEL, False, estado)
    assert anotada.ndim == 3 and mapa is not None and len(pasos) == 7
    assert "Se encontraron" in resumen and "de tu equipo" in resumen and "Balón" in resumen and "En tu media cancha" in resumen
    assert set(analizar.EXPLICACION_PASOS) == {p.nombre.split(".")[0] for p in pasos}


def test_analizar_sin_esquinas_no_dibuja_mapa():
    from volleyvision.ui.paginas import analizar
    sel = ui.valores_por_defecto("voleibol_03.jpg")
    if "voleibol_03.jpg" not in ui.io.listar_muestras():
        pytest.skip("Faltan las fotos descargadas")
    estado = ui.estado_nuevo()
    assert "Sin esquinas" in analizar.estado_esquinas(*sel, estado)
    _, mapa, resumen, _ = analizar.analizar(*sel, False, estado)
    assert mapa is None and "En tu media cancha" not in resumen
