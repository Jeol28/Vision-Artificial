"""Componentes reutilizables de la UI (Streamlit).

La UI no procesa imágenes: solo recoge parámetros, llama al ``core`` y muestra
los resultados.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from volleyvision.core import calibracion as cal
from volleyvision.utils import io, sinteticos
from volleyvision.utils import visualizacion as vis

MAX_LADO = 960
MUESTRA_POR_DEFECTO = "cancha_sintetica.jpg"


# --------------------------------------------------------------------------
# Estilo y textos
# --------------------------------------------------------------------------
def estilos() -> None:
    """Ajustes de estilo sobre el tema definido en ``.streamlit/config.toml``."""
    st.markdown(
        """
        <style>
        h1, h2, h3 { letter-spacing: -0.01em; }
        h1 span.acento, h2 span.acento { color: #FF6B35; }
        div[data-testid="stExpander"] details summary p { font-weight: 600; color: #FF6B35; }
        div[data-testid="stMetricValue"] { color: #F5F5F7; }
        .vv-nota { color: #B8BFCC; font-size: 0.9rem; }
        .vv-orden { background: #243044; border-left: 4px solid #FF6B35; padding: 0.5rem 0.8rem;
                    border-radius: 4px; margin-bottom: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def encabezado(titulo: str, subtitulo: str) -> None:
    st.markdown(f"## {titulo}")
    st.caption(subtitulo)


def que_esta_pasando(texto: str, abierto: bool = False) -> None:
    """Expander con la explicación teórica corta (admite LaTeX con $...$ y $$...$$)."""
    with st.expander("¿Qué está pasando?", expanded=abierto):
        st.markdown(texto)


# --------------------------------------------------------------------------
# Imágenes
# --------------------------------------------------------------------------
def mostrar(img: np.ndarray, titulo: str = "") -> None:
    """Muestra una imagen BGR (o de grises) ocupando el ancho de la columna."""
    st.image(io.bgr_a_rgb(vis.a_uint8(img)), caption=titulo or None, width="stretch")


def lado_a_lado(izq: np.ndarray, der: np.ndarray, titulo_izq: str = "Original", titulo_der: str = "Resultado"):
    c1, c2 = st.columns(2)
    with c1:
        mostrar(izq, titulo_izq)
    with c2:
        mostrar(der, titulo_der)


def cuadricula(imagenes: dict[str, np.ndarray], columnas: int = 3) -> None:
    """Muestra varias imágenes en una cuadrícula ``titulo → imagen``."""
    items = list(imagenes.items())
    for i in range(0, len(items), columnas):
        cols = st.columns(columnas)
        for col, (titulo, img) in zip(cols, items[i:i + columnas]):
            with col:
                mostrar(img, titulo)


def figura(fig) -> None:
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# --------------------------------------------------------------------------
# Selección de imagen / video
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cargar_ruta(ruta: str, max_lado: int) -> np.ndarray:
    return io.cargar_imagen(ruta, max_lado)


@st.cache_data(show_spinner=False)
def _cargar_bytes(datos: bytes, max_lado: int) -> np.ndarray:
    return io.cargar_imagen(datos, max_lado)


@st.cache_data(show_spinner=False)
def _guardar_video_temporal(datos: bytes, sufijo: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=sufijo)
    tmp.write(datos)
    tmp.close()
    return tmp.name


def selector_imagen(clave: str, permitir_video: bool = True,
                    por_defecto: str = MUESTRA_POR_DEFECTO) -> tuple[np.ndarray, str]:
    """Selector de imagen: muestra de ejemplo, archivo subido o frame de un video.

    Returns:
        (imagen BGR, nombre de la escena). El nombre identifica la escena para
        recordar sus esquinas de cancha entre páginas (en un video, todas los frames
        comparten esquinas porque la cámara es fija).
    """
    clave = f"sel_{clave}"  # evita choques con las claves de los widgets de cada página
    fuentes = ["Imagen de ejemplo", "Subir imagen"] + (["Video (elegir frame)"] if permitir_video else [])
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            fuente = st.radio("Fuente", fuentes, key=f"{clave}_fuente", horizontal=False)
        with c2:
            if fuente == "Imagen de ejemplo":
                muestras = io.listar_muestras()
                if not muestras:
                    st.error("No hay muestras. Ejecuta `python scripts/generar_sinteticos.py`.")
                    st.stop()
                nombres = list(muestras)
                idx = nombres.index(por_defecto) if por_defecto in nombres else 0
                nombre = st.selectbox("Imagen", nombres, index=idx, key=f"{clave}_muestra")
                img = _cargar_ruta(str(muestras[nombre]), MAX_LADO)
            elif fuente == "Subir imagen":
                archivo = st.file_uploader("Imagen (JPG/PNG)", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                           key=f"{clave}_archivo")
                if archivo is None:
                    st.info("Sube una imagen para continuar.")
                    st.stop()
                nombre = archivo.name
                img = _cargar_bytes(archivo.getvalue(), MAX_LADO)
            else:
                videos = io.listar_videos()
                opciones = list(videos) + ["Subir video…"]
                eleccion = st.selectbox("Video", opciones, key=f"{clave}_video")
                if eleccion == "Subir video…":
                    archivo = st.file_uploader("Video (MP4/AVI/MOV)", type=["mp4", "avi", "mov", "mkv"],
                                               key=f"{clave}_video_archivo")
                    if archivo is None:
                        st.info("Sube un video para continuar.")
                        st.stop()
                    ruta = _guardar_video_temporal(archivo.getvalue(), Path(archivo.name).suffix)
                    nombre = archivo.name
                else:
                    ruta, nombre = str(videos[eleccion]), eleccion
                n, fps = io.info_video(ruta)
                frame = st.slider(f"Frame (de {n}, {fps:.0f} fps)", 0, max(0, n - 1), min(n - 1, n // 2),
                                  key=f"{clave}_frame")
                img = io.redimensionar_max(io.leer_frame(ruta, frame), MAX_LADO)
    return img, nombre


# --------------------------------------------------------------------------
# Esquinas de la cancha (compartidas entre páginas)
# --------------------------------------------------------------------------
def _esquinas_guardadas() -> dict:
    return st.session_state.setdefault("esquinas", {})


def esquinas_de(nombre: str) -> np.ndarray | None:
    """Esquinas ya marcadas para esta escena (4×2) o ``None``."""
    pts = _esquinas_guardadas().get(nombre)
    return np.array(pts, np.float32) if pts is not None and len(pts) == 4 else None


def esquinas_ejemplo(nombre: str, img: np.ndarray) -> np.ndarray | None:
    """Esquinas reales de las escenas sintéticas (escaladas al tamaño mostrado)."""
    if nombre.startswith("cancha_sintetica") or nombre.startswith("rally_sintetico"):
        h, w = img.shape[:2]
        esc = np.array([w / sinteticos.TAM_ESCENA[0], h / sinteticos.TAM_ESCENA[1]], np.float32)
        return sinteticos.ESQUINAS_ESCENA * esc
    return None


def selector_esquinas(img: np.ndarray, nombre: str, clave: str, ancho: int = 760) -> np.ndarray | None:
    """Marca las 4 esquinas de la cancha haciendo clic sobre la imagen.

    Incluye: orden indicado en pantalla, puntos dibujados, botón de reinicio y
    entrada manual de coordenadas como respaldo.

    Returns:
        Las 4 esquinas (4×2, píxeles de ``img``) o ``None`` si aún faltan.
    """
    guardadas = _esquinas_guardadas()
    puntos: list[list[float]] = list(guardadas.get(nombre, []))
    h, w = img.shape[:2]

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        if len(puntos) < 4:
            st.markdown(f"<div class='vv-orden'>Haz clic en la esquina <b>{len(puntos) + 1}/4: "
                        f"{cal.ORDEN_ESQUINAS[len(puntos)]}</b> de la cancha. Orden: "
                        "sup-izq → sup-der → inf-der → inf-izq.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='vv-orden'>✅ Las 4 esquinas están marcadas.</div>", unsafe_allow_html=True)
    with c2:
        if st.button("↺ Reiniciar puntos", key=f"{clave}_reiniciar", width="stretch"):
            guardadas[nombre] = []
            st.session_state.pop(f"{clave}_ultimo_clic", None)
            st.session_state[f"{clave}_componente_v"] = st.session_state.get(f"{clave}_componente_v", 0) + 1
            st.rerun()
    with c3:
        ejemplo = esquinas_ejemplo(nombre, img)
        if ejemplo is not None and st.button("Usar esquinas de ejemplo", key=f"{clave}_ejemplo", width="stretch"):
            guardadas[nombre] = ejemplo.tolist()
            st.rerun()

    dibujada = vis.dibujar_puntos(img, puntos, vis.NARANJA, numerar=True, cerrar=True)
    valor = None
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates

        version = st.session_state.get(f"{clave}_componente_v", 0)
        valor = streamlit_image_coordinates(io.bgr_a_rgb(dibujada), width=min(ancho, w),
                                            key=f"{clave}_clic_{version}")
    except Exception as e:  # el componente puede fallar en algunos navegadores
        st.warning(f"No se pudo cargar el selector por clic ({e}). Usa la entrada manual.")
        mostrar(dibujada)
    if valor is not None and len(puntos) < 4:
        marca = (valor.get("x"), valor.get("y"), valor.get("unix_time"))
        if st.session_state.get(f"{clave}_ultimo_clic") != marca:
            st.session_state[f"{clave}_ultimo_clic"] = marca
            esc_x = w / valor["width"]
            esc_y = h / valor["height"]
            puntos.append([valor["x"] * esc_x, valor["y"] * esc_y])
            guardadas[nombre] = puntos
            st.rerun()

    with st.expander("Escribir coordenadas a mano (respaldo)"):
        st.caption(f"Coordenadas en píxeles de la imagen ({w}×{h}).")
        base = puntos + [[0.0, 0.0]] * (4 - len(puntos)) if len(puntos) < 4 else puntos
        nuevos = []
        for i, etiqueta in enumerate(cal.ORDEN_ESQUINAS):
            a, b = st.columns(2)
            x = a.number_input(f"{etiqueta}: x", 0.0, float(w), float(base[i][0]), 1.0, key=f"{clave}_mx{i}")
            y = b.number_input(f"{etiqueta}: y", 0.0, float(h), float(base[i][1]), 1.0, key=f"{clave}_my{i}")
            nuevos.append([x, y])
        if st.button("Usar estas coordenadas", key=f"{clave}_manual"):
            guardadas[nombre] = nuevos
            st.rerun()

    return esquinas_de(nombre)


# --------------------------------------------------------------------------
# Escena sintética: verdad conocida
# --------------------------------------------------------------------------
SEMILLAS_MUESTRA = {"cancha_sintetica.jpg": 0, "cancha_sintetica_2.jpg": 5}


@st.cache_data(show_spinner=False)
def verdad_muestra(nombre: str):
    """Cajas reales de los jugadores si la imagen es una de las escenas sintéticas."""
    if nombre not in SEMILLAS_MUESTRA:
        return None
    escena = sinteticos.generar_cancha_sintetica(SEMILLAS_MUESTRA[nombre])
    return [j["caja"] for j in escena.jugadores]


def a_gris(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
