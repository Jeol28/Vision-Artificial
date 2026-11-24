import streamlit as st

from volleyvision.core import transformers_models as tm
from volleyvision.ui import componentes as ui
from volleyvision.utils import io

st.markdown("# 🏐 Volley<span style='color:#FF6B35'>Vision</span>", unsafe_allow_html=True)
st.markdown("**Proyecto final – Visión Artificial** · Pontificia Universidad Javeriana  \n"
            "Autor: **Jorge Enrique Olaya Liévano**")

st.markdown(
    """
Aplicación que analiza fotos y video de un partido de voleibol con un **pipeline de visión por
computador**. Cada tema del curso es una etapa del pipeline y, al final, se comparan las técnicas
clásicas con modelos **transformer** de Hugging Face (solo inferencia, pesos públicos).
"""
)

c1, c2 = st.columns([3, 2])
with c1:
    st.markdown("#### Recorrido")
    st.markdown(
        """
| Página | Qué hace con el voleibol |
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
    )
with c2:
    st.markdown("#### Pipeline")
    st.markdown(
        """
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
    )
    st.markdown("#### Entorno")
    st.write(f"Dispositivo para los transformers: **{tm.elegir_dispositivo()}**")
    st.write(f"Imágenes de ejemplo: **{len(io.listar_muestras())}** · Videos: **{len(io.listar_videos())}**")
    if not (io.DIR_MUESTRAS / "descargadas").exists():
        st.caption("Para fotos reales ejecuta `python scripts/download_samples.py`.")

ui.que_esta_pasando(
    """
**Arquitectura en capas.** La lógica de visión está en `volleyvision/core/` como funciones puras
(imagen de entrada → imagen/datos de salida), sin Streamlit. Esta UI (`volleyvision/ui/`) solo recoge
parámetros, llama al core y muestra resultados. `volleyvision/utils/` contiene la E/S, el dibujo y los
datos sintéticos. Por eso el core se prueba con pytest y se puede reutilizar fuera de la app.

Los modelos transformer se descargan la primera vez que se usan y se cargan **una sola vez** con
`st.cache_resource`.
""",
    abierto=True,
)
