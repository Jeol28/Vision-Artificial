"""Punto de entrada de la UI.

Ejecutar desde la raíz del repositorio:

    streamlit run volleyvision/ui/app.py
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import streamlit as st  # noqa: E402

from volleyvision.ui.componentes import estilos  # noqa: E402

st.set_page_config(page_title="VolleyVision", page_icon="🏐", layout="wide")
estilos()

PAGINAS = Path(__file__).parent / "paginas"

navegacion = st.navigation({
    "VolleyVision": [
        st.Page(PAGINAS / "0_inicio.py", title="Inicio", icon=":material/home:", default=True),
    ],
    "Temas del curso": [
        st.Page(PAGINAS / "1_calibracion.py", title="1 · Calibración", icon=":material/photo_camera:"),
        st.Page(PAGINAS / "2_filtros.py", title="2 · Filtros", icon=":material/blur_on:"),
        st.Page(PAGINAS / "3_histogramas.py", title="3 · Histogramas", icon=":material/bar_chart:"),
        st.Page(PAGINAS / "4_umbralizacion.py", title="4 · Umbralización", icon=":material/contrast:"),
        st.Page(PAGINAS / "5_morfologia.py", title="5 · Morfología", icon=":material/category:"),
        st.Page(PAGINAS / "6_segmentacion.py", title="6 · Segmentación", icon=":material/texture:"),
        st.Page(PAGINAS / "7_caracterizacion.py", title="7 · Caracterización", icon=":material/straighten:"),
    ],
    "Integración": [
        st.Page(PAGINAS / "8_transformers.py", title="8 · Transformers", icon=":material/neurology:"),
        st.Page(PAGINAS / "9_pipeline.py", title="9 · Pipeline", icon=":material/account_tree:"),
    ],
})

with st.sidebar:
    st.markdown("**VolleyVision** · Visión Artificial")
    st.caption("Jorge Enrique Olaya Liévano · Pontificia Universidad Javeriana")

navegacion.run()
