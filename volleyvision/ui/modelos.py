"""Carga de modelos transformer con ``st.cache_resource``.

Cada modelo se descarga (la primera vez) y se carga en memoria **una sola vez**
por proceso de Streamlit; todas las páginas y sesiones lo reutilizan.
"""

from __future__ import annotations

import streamlit as st

from volleyvision.core import transformers_models as tm


@st.cache_resource(show_spinner=False)
def detector(id_hf: str) -> tm.ModeloCargado:
    modelo = tm.cargar_detector(id_hf)
    tm.calentar(modelo)  # la primera inferencia es más lenta: no debe contar en la comparación
    return modelo


@st.cache_resource(show_spinner=False)
def segformer() -> tm.ModeloCargado:
    return tm.cargar_segformer()


@st.cache_resource(show_spinner=False)
def sam(id_hf: str) -> tm.ModeloCargado:
    return tm.cargar_sam(id_hf)


def obtener_detector(nombre: str) -> tm.ModeloCargado:
    id_hf = tm.MODELOS_DETECCION[nombre]
    with st.spinner(f"Cargando {id_hf} (la primera vez se descarga de Hugging Face)…"):
        return detector(id_hf)


def obtener_segformer() -> tm.ModeloCargado:
    with st.spinner(f"Cargando {tm.MODELO_SEGFORMER}…"):
        return segformer()


def obtener_sam(nombre: str) -> tm.ModeloCargado:
    id_hf = tm.MODELOS_SAM[nombre]
    with st.spinner(f"Cargando {id_hf}…"):
        return sam(id_hf)


def selector_detector(clave: str) -> str:
    """Selectbox de detector con RT-DETR por defecto."""
    return st.selectbox("Detector transformer", list(tm.MODELOS_DETECCION), index=0, key=clave)
