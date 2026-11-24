"""Construcción de la interfaz de VolleyVision (Gradio).

Ejecutar desde la raíz del repositorio::

    python app.py

En Hugging Face Spaces (ZeroGPU) se ejecuta el mismo ``app.py`` de la raíz.
"""

from __future__ import annotations

import gradio as gr

from volleyvision.ui import componentes as ui
from volleyvision.ui.paginas import (p0_inicio, p1_calibracion, p2_filtros, p3_histogramas, p4_umbralizacion,
                                     p5_morfologia, p6_segmentacion, p7_caracterizacion, p8_transformers,
                                     p9_pipeline)

PAGINAS = [
    ("Inicio", p0_inicio),
    ("1 · Calibración", p1_calibracion),
    ("2 · Filtros", p2_filtros),
    ("3 · Histogramas", p3_histogramas),
    ("4 · Umbralización", p4_umbralizacion),
    ("5 · Morfología", p5_morfologia),
    ("6 · Segmentación", p6_segmentacion),
    ("7 · Caracterización", p7_caracterizacion),
    ("8 · Transformers", p8_transformers),
    ("9 · Pipeline", p9_pipeline),
]


def construir_app() -> gr.Blocks:
    """Una pestaña por tema del curso; el estado de la sesión (esquinas y calibración) se comparte."""
    with gr.Blocks(title="VolleyVision") as demo:
        estado = gr.State(ui.estado_nuevo())
        with gr.Tabs():
            for titulo, pagina in PAGINAS:
                with gr.Tab(titulo):
                    pagina.construir(estado)
        gr.Markdown("<span class='vv-nota'>**VolleyVision** · Visión Artificial · Jorge Enrique Olaya Liévano · "
                    "Pontificia Universidad Javeriana</span>")
    return demo


def lanzar(demo: gr.Blocks, **kw) -> None:
    demo.queue(default_concurrency_limit=4).launch(theme=ui.tema(), css=ui.CSS, js=ui.JS_MODO_OSCURO,
                                                   max_file_size="200mb", **kw)
