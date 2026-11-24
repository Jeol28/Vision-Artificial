"""Página de inicio: presentación, recorrido y entorno."""

import gradio as gr

from volleyvision.core import transformers_models as tm
from volleyvision.ui import componentes as ui
from volleyvision.ui import modelos
from volleyvision.utils import io

RECORRIDO = """
#### Recorrido

| Pestaña | Qué hace con el voleibol |
|---|---|
| 1 · Calibración | Matriz K, distorsión, error de reproyección y vista cenital desde la cámara trasera |
| 2 · Filtros | Suavizado, bordes, convolución manual y realce de las líneas |
| 3 · Histogramas | RGB/HSV, ecualización, CLAHE y comparación de uniformes |
| 4 · Umbralización | Global, Otsu manual, adaptativo y rangos HSV (balón, cancha) |
| 5 · Morfología | Limpieza de las máscaras del balón y de los jugadores |
| 6 · Segmentación | Contornos, watershed, k-means, SegFormer y SAM |
| 7 · Caracterización | Descriptores, "¿es balón?" y proyección a la cancha |
| 8 · Transformers | RT-DETR / DETR y comparación clásico vs. transformer |
| 9 · Pipeline | Todas las etapas encadenadas sobre una imagen o frame de video |
"""

PIPELINE = """
#### Pipeline

```
Imagen / frame
  │ 1. Corrección de distorsión
  │ 2. CLAHE
  │ 3. Bilateral + Canny + Hough
  │ 4. Umbral (Otsu, HSV) + morfología
  │ 5. Segmentación clásica ┃ transformer
  │ 6. Caracterización + equipos
  ▼ 7. Mapa cenital (homografía)
```
"""


def entorno() -> str:
    disp = "GPU de ZeroGPU (Hugging Face Spaces)" if modelos.EN_ZEROGPU else tm.elegir_dispositivo()
    texto = (f"#### Entorno\n\nDispositivo para los transformers: **{disp}**  \n"
             f"Imágenes de ejemplo: **{len(io.listar_muestras())}** · Videos: **{len(io.listar_videos())}**")
    if not (io.DIR_MUESTRAS / "descargadas").exists():
        texto += "\n\n<span class='vv-nota'>Para fotos reales ejecuta `python scripts/download_samples.py`.</span>"
    return texto


def construir(estado: gr.State) -> None:
    gr.Markdown("# 🏐 Volley<span class='acento'>Vision</span>\n"
                "**Proyecto final – Visión Artificial** · Pontificia Universidad Javeriana  \n"
                "Autor: **Jorge Enrique Olaya Liévano**")
    gr.Markdown("Aplicación que analiza fotos y video de un partido de voleibol con un **pipeline de visión por "
                "computador**, con la cámara detrás de la línea de fondo propia. Cada tema del curso es una etapa "
                "del pipeline y, al final, se comparan las técnicas clásicas con modelos **transformer** de "
                "Hugging Face (solo inferencia, pesos públicos).")
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(RECORRIDO)
        with gr.Column(scale=2):
            gr.Markdown(PIPELINE)
            gr.Markdown(entorno())
    ui.que_esta_pasando(
        """
**Arquitectura en capas.** La lógica de visión está en `volleyvision/core/` como funciones puras
(imagen de entrada → imagen/datos de salida), sin Gradio. Esta UI (`volleyvision/ui/`) solo recoge
parámetros, llama al core y muestra resultados. `volleyvision/utils/` contiene la E/S, el dibujo y los
datos sintéticos. Por eso el core se prueba con pytest y se puede reutilizar fuera de la app.

Los modelos transformer se cargan **una sola vez** por proceso (`ui/modelos.py`). En Hugging Face Spaces
con **ZeroGPU** se cargan al arrancar y la inferencia corre en funciones `@spaces.GPU`, que reciben una
GPU solo mientras se ejecutan.
""",
        abierto=True,
    )
