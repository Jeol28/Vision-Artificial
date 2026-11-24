"""Construcción de la interfaz de VolleyVision (Gradio).

Ejecutar desde la raíz del repositorio::

    python app.py

En Hugging Face Spaces (ZeroGPU) se ejecuta el mismo ``app.py`` de la raíz.

La interfaz tiene tres secciones:

* **Analizar partido**: modo guiado en 3 pasos para quien no es experto.
* **Temas del curso**: una pestaña por tema, con controles y teoría.
* **Acerca del proyecto**: autor, recorrido, pipeline y arquitectura.
"""

from __future__ import annotations

import gradio as gr

from volleyvision.ui import componentes as ui
from volleyvision.ui.paginas import (analizar, p0_inicio, p1_calibracion, p2_filtros, p3_histogramas,
                                     p4_umbralizacion, p5_morfologia, p6_segmentacion, p7_caracterizacion,
                                     p8_transformers, p9_pipeline)

TEMAS = [
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
    """Las secciones comparten el estado de la sesión (esquinas marcadas y calibración)."""
    with gr.Blocks(title="VolleyVision") as demo:
        estado = gr.State(ui.estado_nuevo())
        with gr.Tabs():
            with gr.Tab("🏐 Analizar partido"):
                analizar.construir(estado)
            with gr.Tab("📚 Temas del curso"):
                ui.md("Cada pestaña es un tema de la materia aplicado al voleibol. Arriba de cada una hay una "
                      "explicación corta; los controles que no hace falta tocar están en **⚙️ Opciones "
                      "avanzadas**, y la teoría con fórmulas en **¿Qué está pasando?**.")
                with gr.Tabs():
                    for titulo, pagina in TEMAS:
                        with gr.Tab(titulo):
                            pagina.construir(estado)
            with gr.Tab("ℹ️ Acerca del proyecto"):
                p0_inicio.construir(estado)
        gr.Markdown("<span class='vv-nota'>**VolleyVision** · Visión Artificial · Jorge Enrique Olaya Liévano · "
                    "Pontificia Universidad Javeriana</span>")
    return demo


def lanzar(demo: gr.Blocks, **kw) -> None:
    demo.queue(default_concurrency_limit=4).launch(theme=ui.tema(), css=ui.CSS, js=ui.JS_MODO_OSCURO,
                                                   max_file_size="200mb", **kw)
