"""Componentes reutilizables de la UI (Gradio).

La UI no procesa imágenes: solo recoge parámetros, llama al ``core`` y muestra
los resultados. Las imágenes viajan en BGR dentro del proyecto y se pasan a RGB
solo al mostrarlas.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from volleyvision.core import calibracion as cal
from volleyvision.utils import io, sinteticos
from volleyvision.utils import visualizacion as vis

MAX_LADO = 960
MUESTRA_POR_DEFECTO = "cancha_sintetica.jpg"
FUENTES = ("Imagen de ejemplo", "Subir imagen", "Video (elegir frame)")
SUBIR_VIDEO = "Subir video…"
LATEX = [{"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False}]

# Estado de cada sesión: esquinas marcadas por escena y la calibración de la cámara.
ESTADO_INICIAL = {"esquinas": {}, "calibracion": None, "calibracion_imagenes": None,
                  "calibracion_patron": None}

CSS = """
.gradio-container { max-width: 1400px !important; }
h1 .acento, h2 .acento { color: #FF6B35; }
.vv-orden { background: #243044; border-left: 4px solid #FF6B35; padding: .5rem .8rem; border-radius: 4px; }
.vv-nota { color: #B8BFCC; font-size: .9rem; }
.vv-metricas { display: flex; flex-wrap: wrap; gap: .6rem; margin: .3rem 0; }
.vv-metrica { background: #243044; border-radius: 6px; padding: .5rem .8rem; min-width: 9rem; flex: 1; }
.vv-metrica .etq { color: #B8BFCC; font-size: .8rem; }
.vv-metrica .val { color: #F5F5F7; font-size: 1.35rem; font-weight: 600; }
.vv-explicacion > .label-wrap span { color: #FF6B35 !important; font-weight: 600; }
"""


def tema() -> gr.themes.Base:
    """Tema oscuro con la paleta del proyecto (grafito, naranja y gris claro)."""
    naranja = gr.themes.Color(c50="#FFF1EB", c100="#FFE0D2", c200="#FFC2A6", c300="#FFA37A", c400="#FF8757",
                              c500="#FF6B35", c600="#E85A26", c700="#C2481D", c800="#9C3917", c900="#7A2D13",
                              c950="#4D1C0C")
    grafito = gr.themes.Color(c50="#F5F5F7", c100="#E3E6EC", c200="#C4CAD6", c300="#9AA3B5", c400="#6E7890",
                              c500="#4A5468", c600="#36405A", c700="#2B3549", c800="#243044", c900="#1A2332",
                              c950="#121926")
    return gr.themes.Base(primary_hue=naranja, secondary_hue=naranja, neutral_hue=grafito).set(
        body_background_fill_dark="#1A2332", block_background_fill_dark="#1F2A3C",
        background_fill_primary_dark="#1A2332", background_fill_secondary_dark="#243044",
        body_text_color_dark="#F5F5F7", border_color_primary_dark="#36405A",
        button_primary_background_fill_dark="#FF6B35", button_primary_background_fill_hover_dark="#E85A26",
        button_primary_text_color_dark="#1A2332", slider_color_dark="#FF6B35",
        color_accent="#FF6B35", color_accent_soft_dark="#243044")


# Fuerza el modo oscuro (la paleta está pensada sobre fondo grafito).
JS_MODO_OSCURO = "() => { document.body.classList.add('dark'); }"


# --------------------------------------------------------------------------
# Textos y salidas
# --------------------------------------------------------------------------
def encabezado(titulo: str, subtitulo: str) -> None:
    gr.Markdown(f"## {titulo}\n<span class='vv-nota'>{subtitulo}</span>")


def que_esta_pasando(texto: str, abierto: bool = False) -> None:
    """Acordeón con la explicación teórica corta (admite LaTeX con $...$ y $$...$$)."""
    with gr.Accordion("¿Qué está pasando?", open=abierto, elem_classes="vv-explicacion"):
        gr.Markdown(texto, latex_delimiters=LATEX)


def md(texto: str = "", **kw) -> gr.Markdown:
    return gr.Markdown(texto, latex_delimiters=LATEX, **kw)


def rgb(img: np.ndarray) -> np.ndarray:
    """Imagen BGR (o de grises, o float) lista para mostrar en Gradio."""
    return io.bgr_a_rgb(vis.a_uint8(img))


def galeria(imagenes: dict[str, np.ndarray]) -> list[tuple[np.ndarray, str]]:
    """``titulo → imagen BGR`` al formato de ``gr.Gallery``."""
    return [(rgb(img), titulo) for titulo, img in imagenes.items()]


def nueva_galeria(columnas: int = 3, label: str = "Resultados", height="auto", value=None) -> gr.Gallery:
    return gr.Gallery(value=value, label=label, columns=columnas, height=height, object_fit="contain", show_label=False,
                      allow_preview=True)


def imagen_salida(label: str = "", **kw) -> gr.Image:
    return gr.Image(label=label or None, interactive=False, format="png", **kw)


def figura(fig):
    """Cierra la figura en pyplot (para no acumular memoria) y la devuelve para ``gr.Plot``."""
    plt.close(fig)
    return fig


def metricas(valores: dict[str, object]) -> str:
    """Tarjetas de métricas en HTML ``etiqueta → valor``."""
    tarjetas = "".join(f"<div class='vv-metrica'><div class='etq'>{k}</div><div class='val'>{v}</div></div>"
                       for k, v in valores.items())
    return f"<div class='vv-metricas'>{tarjetas}</div>"


def a_gris(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------
# Selección de imagen / video
# --------------------------------------------------------------------------
@dataclass
class SelectorImagen:
    """Componentes del selector de imagen de una página."""

    fuente: gr.Radio
    muestra: gr.Dropdown
    subida: gr.Image
    video: gr.Dropdown
    video_subido: gr.File
    frame: gr.Slider

    @property
    def entradas(self) -> list:
        return [self.fuente, self.muestra, self.subida, self.video, self.video_subido, self.frame]


def _ruta_video(video: str | None, video_subido) -> str | None:
    if video == SUBIR_VIDEO:
        if video_subido is None:
            return None
        return video_subido if isinstance(video_subido, str) else video_subido.name
    videos = io.listar_videos()
    return str(videos[video]) if video in videos else None


def _actualizar_frame(video, video_subido):
    ruta = _ruta_video(video, video_subido)
    if ruta is None:
        return gr.Slider(maximum=1, value=0, label="Frame")
    n, fps = io.info_video(ruta)
    return gr.Slider(minimum=0, maximum=max(1, n - 1), value=min(n - 1, n // 2), step=1,
                     label=f"Frame (de {n}, {fps:.0f} fps)")


def _visibilidad(fuente, video):
    es_video = fuente == FUENTES[2]
    return (gr.Dropdown(visible=fuente == FUENTES[0]), gr.Image(visible=fuente == FUENTES[1]),
            gr.Dropdown(visible=es_video), gr.File(visible=es_video and video == SUBIR_VIDEO),
            gr.Slider(visible=es_video))


def selector_imagen(por_defecto: str = MUESTRA_POR_DEFECTO, permitir_video: bool = True) -> SelectorImagen:
    """Selector de imagen: muestra de ejemplo, archivo subido o frame de un video.

    Las muestras incluyen la escena sintética (con verdad conocida) y las fotos
    reales descargadas con ``scripts/download_samples.py``.
    """
    muestras = list(io.listar_muestras())
    videos = list(io.listar_videos()) + [SUBIR_VIDEO]
    valor_muestra = por_defecto if por_defecto in muestras else (muestras[0] if muestras else None)
    valor_video = videos[0]
    n_frames = io.info_video(io.listar_videos()[valor_video])[0] if valor_video != SUBIR_VIDEO else 2
    with gr.Group():
        with gr.Row(equal_height=False):
            fuente = gr.Radio(list(FUENTES if permitir_video else FUENTES[:2]), value=FUENTES[0], label="Fuente",
                              scale=1)
            with gr.Column(scale=2):
                muestra = gr.Dropdown(muestras, value=valor_muestra, label="Imagen")
                subida = gr.Image(type="filepath", label="Imagen (JPG/PNG)", sources=["upload", "clipboard"],
                                  visible=False, height=220)
                video = gr.Dropdown(videos, value=valor_video, label="Video", visible=False)
                video_subido = gr.File(label="Video (MP4/AVI/MOV)", file_types=["video"], visible=False)
                frame = gr.Slider(0, max(1, n_frames - 1), value=n_frames // 2, step=1, label="Frame",
                                  visible=False)
    sel = SelectorImagen(fuente, muestra, subida, video, video_subido, frame)
    fuente.change(_visibilidad, [fuente, video], [muestra, subida, video, video_subido, frame])
    video.change(_visibilidad, [fuente, video], [muestra, subida, video, video_subido, frame])
    gr.on([video.change, video_subido.change], _actualizar_frame, [video, video_subido], frame)
    return sel


def resolver_imagen(fuente, muestra, subida, video, video_subido, frame) -> tuple[np.ndarray, str]:
    """Convierte los valores del selector en ``(imagen BGR, nombre de la escena)``.

    El nombre identifica la escena para recordar sus esquinas de cancha entre
    páginas; en un video todos los frames comparten esquinas (cámara fija).
    """
    if fuente == FUENTES[1]:
        if not subida:
            raise gr.Error("Sube una imagen para continuar.")
        img = io.cargar_imagen(subida, MAX_LADO)
        nombre = Path(subida).name
        if nombre.startswith("image."):  # imagen pegada del portapapeles: nombre genérico
            nombre = "subida_" + hashlib.md5(img.tobytes()).hexdigest()[:8]
        return img, nombre
    if fuente == FUENTES[2]:
        ruta = _ruta_video(video, video_subido)
        if ruta is None:
            raise gr.Error("Elige o sube un video para continuar.")
        nombre = Path(video_subido if video == SUBIR_VIDEO and isinstance(video_subido, str) else ruta).name
        return io.redimensionar_max(io.leer_frame(ruta, int(frame or 0)), MAX_LADO), nombre
    muestras = io.listar_muestras()
    if not muestras:
        raise gr.Error("No hay muestras. Ejecuta `python scripts/generar_sinteticos.py`.")
    nombre = muestra if muestra in muestras else next(iter(muestras))
    return io.cargar_imagen(muestras[nombre], MAX_LADO), nombre


def valores_por_defecto(por_defecto: str = MUESTRA_POR_DEFECTO) -> tuple:
    """Valores del selector con la imagen de ejemplo (útil en pruebas)."""
    return FUENTES[0], por_defecto, None, None, None, 0


# --------------------------------------------------------------------------
# Esquinas de la cancha (compartidas entre páginas por el estado de la sesión)
# --------------------------------------------------------------------------
def estado_nuevo() -> dict:
    return copy.deepcopy(ESTADO_INICIAL)


def esquinas_de(estado: dict, nombre: str) -> np.ndarray | None:
    """Esquinas ya marcadas para esta escena (4×2) o ``None``."""
    pts = (estado or {}).get("esquinas", {}).get(nombre)
    return np.array(pts, np.float32) if pts is not None and len(pts) == 4 else None


def esquinas_ejemplo(nombre: str, img: np.ndarray) -> np.ndarray | None:
    """Esquinas reales de las escenas sintéticas (escaladas al tamaño mostrado)."""
    if nombre.startswith("cancha_sintetica") or nombre.startswith("rally_sintetico"):
        h, w = img.shape[:2]
        esc = np.array([w / sinteticos.TAM_ESCENA[0], h / sinteticos.TAM_ESCENA[1]], np.float32)
        return sinteticos.ESQUINAS_ESCENA * esc
    return None


def _instruccion(n: int) -> str:
    if n >= 4:
        return "<div class='vv-orden'>✅ Las 4 esquinas están marcadas.</div>"
    return (f"<div class='vv-orden'>Haz clic en la esquina <b>{n + 1}/4: {cal.ORDEN_ESQUINAS[n]}</b> de tu media "
            "cancha. Orden: sup-izq → sup-der (bajo la red) → inf-der → inf-izq (tu línea de fondo).</div>")


def _vista_esquinas(img: np.ndarray, nombre: str, estado: dict):
    puntos = (estado or {}).get("esquinas", {}).get(nombre, [])
    dibujada = vis.dibujar_puntos(img, puntos, vis.NARANJA, numerar=True, cerrar=True)
    return rgb(dibujada), _instruccion(len(puntos))


@dataclass
class SelectorEsquinas:
    imagen: gr.Image
    instruccion: gr.HTML


def selector_esquinas(sel: SelectorImagen, estado: gr.State) -> SelectorEsquinas:
    """Marca las 4 esquinas de la media cancha propia (cámara trasera) con clics sobre la imagen.

    Incluye el orden indicado en pantalla, los puntos dibujados, un botón para
    reiniciar, las esquinas de ejemplo de la escena sintética y la entrada manual
    de coordenadas como respaldo. Las esquinas quedan en ``estado`` para las
    páginas de Caracterización y Pipeline.
    """
    instruccion = gr.HTML(_instruccion(0))
    imagen = gr.Image(label="Haz clic sobre las esquinas", interactive=False, format="png",
                      buttons=["fullscreen"])
    with gr.Row():
        reiniciar = gr.Button("↺ Reiniciar puntos", size="sm")
        ejemplo = gr.Button("Usar esquinas de ejemplo", size="sm")
    with gr.Accordion("Escribir coordenadas a mano (respaldo)", open=False):
        md("<span class='vv-nota'>Coordenadas en píxeles de la imagen mostrada.</span>")
        numeros = []
        for etiqueta in cal.ORDEN_ESQUINAS:
            with gr.Row():
                numeros.append(gr.Number(0, label=f"{etiqueta}: x", minimum=0))
                numeros.append(gr.Number(0, label=f"{etiqueta}: y", minimum=0))
        manual = gr.Button("Usar estas coordenadas", size="sm")

    entradas = sel.entradas

    def refrescar(*args):
        *vals, est = args
        img, nombre = resolver_imagen(*vals)
        return _vista_esquinas(img, nombre, est)

    # Gradio solo inyecta ``gr.SelectData`` en parámetros con nombre (no en ``*args``).
    def clic(fuente, muestra, subida, video, video_subido, frame, est, evt: gr.SelectData):
        img, nombre = resolver_imagen(fuente, muestra, subida, video, video_subido, frame)
        est = copy.deepcopy(est or estado_nuevo())
        puntos = est["esquinas"].setdefault(nombre, [])
        if len(puntos) < 4:
            puntos.append([float(evt.index[0]), float(evt.index[1])])
        return (*_vista_esquinas(img, nombre, est), est)

    def borrar(*args):
        *vals, est = args
        img, nombre = resolver_imagen(*vals)
        est = copy.deepcopy(est or estado_nuevo())
        est["esquinas"][nombre] = []
        return (*_vista_esquinas(img, nombre, est), est)

    def usar_ejemplo(*args):
        *vals, est = args
        img, nombre = resolver_imagen(*vals)
        ej = esquinas_ejemplo(nombre, img)
        if ej is None:
            raise gr.Error("Solo las escenas sintéticas tienen esquinas de ejemplo: márcalas con clics.")
        est = copy.deepcopy(est or estado_nuevo())
        est["esquinas"][nombre] = ej.tolist()
        return (*_vista_esquinas(img, nombre, est), est)

    def usar_manual(*args):
        vals, est, coords = args[:6], args[6], args[7:]
        img, nombre = resolver_imagen(*vals)
        est = copy.deepcopy(est or estado_nuevo())
        est["esquinas"][nombre] = [[float(coords[2 * i] or 0), float(coords[2 * i + 1] or 0)] for i in range(4)]
        return (*_vista_esquinas(img, nombre, est), est)

    # Sin triggers: se ejecuta al cargar la app y cada vez que cambia una entrada o el estado.
    gr.on(None, refrescar, entradas + [estado], [imagen, instruccion], trigger_mode="always_last")
    imagen.select(clic, entradas + [estado], [imagen, instruccion, estado])
    reiniciar.click(borrar, entradas + [estado], [imagen, instruccion, estado])
    ejemplo.click(usar_ejemplo, entradas + [estado], [imagen, instruccion, estado])
    manual.click(usar_manual, entradas + [estado] + numeros, [imagen, instruccion, estado])
    return SelectorEsquinas(imagen, instruccion)


# --------------------------------------------------------------------------
# Escena sintética: verdad conocida
# --------------------------------------------------------------------------
SEMILLAS_MUESTRA = {"cancha_sintetica.jpg": 0, "cancha_sintetica_2.jpg": 5}
_VERDAD: dict[str, list] = {}


def verdad_muestra(nombre: str):
    """Cajas reales de los jugadores si la imagen es una de las escenas sintéticas."""
    if nombre not in SEMILLAS_MUESTRA:
        return None
    if nombre not in _VERDAD:
        escena = sinteticos.generar_cancha_sintetica(SEMILLAS_MUESTRA[nombre])
        _VERDAD[nombre] = [j["caja"] for j in escena.jugadores]
    return _VERDAD[nombre]
