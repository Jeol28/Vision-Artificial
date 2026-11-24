"""Smoke tests de la UI: cada página se ejecuta sin excepciones (streamlit.testing)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGINAS = Path(__file__).resolve().parents[1] / "volleyvision" / "ui" / "paginas"
SIN_MODELOS = ["0_inicio", "1_calibracion", "2_filtros", "3_histogramas", "4_umbralizacion", "5_morfologia",
               "6_segmentacion", "7_caracterizacion"]


def _ejecutar(nombre: str, timeout: int = 120) -> AppTest:
    at = AppTest.from_file(str(PAGINAS / f"{nombre}.py"), default_timeout=timeout)
    at.run()
    assert not at.exception, f"{nombre}: {at.exception}"
    return at


@pytest.mark.parametrize("pagina", SIN_MODELOS)
def test_pagina_carga_sin_errores(pagina):
    _ejecutar(pagina)


def test_app_principal():
    at = AppTest.from_file(str(PAGINAS.parent / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception


def test_calibracion_boton():
    at = _ejecutar("1_calibracion")
    at.button[0].click().run()
    assert not at.exception
    assert any("reproyección" in m.label for m in at.metric)


def test_pipeline_clasico_desde_ui():
    at = _ejecutar("9_pipeline")
    at.checkbox(key="pip_det").uncheck().run()
    boton = next(b for b in at.button if "Ejecutar" in b.label)
    boton.click().run()
    assert not at.exception
    assert any("Tiempo total" in m.label for m in at.metric)


@pytest.mark.slow
def test_pagina_transformers():
    at = _ejecutar("8_transformers", timeout=600)
    assert any("Personas" in m.label for m in at.metric)
