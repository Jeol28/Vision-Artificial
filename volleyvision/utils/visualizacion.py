"""Funciones de dibujo y gráficas reutilizadas por el core y la UI.

Paleta del proyecto (en BGR para OpenCV):

* naranja ``#FF6B35``: acentos;
* verde ``#00C896`` y rojo ``#FF4C4C``: solo para datos (acierto / error,
  equipo A / equipo B, balón / no balón).
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Colores en BGR
NARANJA = (53, 107, 255)
VERDE = (150, 200, 0)
ROJO = (76, 76, 255)
GRAFITO = (50, 35, 26)
NEUTRO = (247, 245, 245)
# Colores en hex para Matplotlib
HEX = {"naranja": "#FF6B35", "verde": "#00C896", "rojo": "#FF4C4C",
       "grafito": "#1A2332", "neutro": "#F5F5F7", "panel": "#243044"}


def a_uint8(img: np.ndarray) -> np.ndarray:
    """Normaliza cualquier arreglo (float, negativos, etc.) al rango 0–255 ``uint8``."""
    if img.dtype == np.uint8:
        return img
    img = np.asarray(img, dtype=np.float64)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-12:
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def a_bgr(img: np.ndarray) -> np.ndarray:
    """Asegura una imagen de 3 canales BGR ``uint8``."""
    img = a_uint8(img)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()


def dibujar_cajas(img: np.ndarray, cajas: Iterable[dict], grosor: int = 2) -> np.ndarray:
    """Dibuja cajas ``{"caja": (x1, y1, x2, y2), "etiqueta": str, "color": bgr}``."""
    salida = a_bgr(img)
    for c in cajas:
        x1, y1, x2, y2 = (int(round(v)) for v in c["caja"])
        color = c.get("color", NARANJA)
        cv2.rectangle(salida, (x1, y1), (x2, y2), color, grosor)
        texto = c.get("etiqueta")
        if texto:
            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            y_txt = max(th + 4, y1)
            cv2.rectangle(salida, (x1, y_txt - th - 4), (x1 + tw + 4, y_txt), color, -1)
            cv2.putText(salida, texto, (x1 + 2, y_txt - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, GRAFITO, 1, cv2.LINE_AA)
    return salida


def superponer_mascara(img: np.ndarray, mascara: np.ndarray, color=NARANJA,
                       alfa: float = 0.5) -> np.ndarray:
    """Pinta la máscara binaria sobre la imagen con transparencia ``alfa``."""
    salida = a_bgr(img)
    m = mascara.astype(bool)
    capa = np.zeros_like(salida)
    capa[:] = color
    salida[m] = (alfa * capa[m] + (1 - alfa) * salida[m]).astype(np.uint8)
    return salida


def colorear_etiquetas(etiquetas: np.ndarray, semilla: int = 7) -> np.ndarray:
    """Asigna un color aleatorio (pero reproducible) a cada etiqueta; 0 → negro."""
    rng = np.random.default_rng(semilla)
    n = int(etiquetas.max()) + 1
    paleta = rng.integers(60, 255, size=(max(n, 1), 3), dtype=np.uint8)
    paleta[0] = 0
    return paleta[np.clip(etiquetas, 0, None)]


def dibujar_puntos(img: np.ndarray, puntos, color=NARANJA, numerar: bool = True,
                   cerrar: bool = False) -> np.ndarray:
    """Dibuja puntos (y opcionalmente el polígono que forman) con su número de orden."""
    salida = a_bgr(img)
    pts = [tuple(int(round(v)) for v in p) for p in puntos]
    radio = max(4, salida.shape[1] // 160)
    if cerrar and len(pts) >= 2:
        cv2.polylines(salida, [np.array(pts, dtype=np.int32)], len(pts) == 4, color, 2)
    for i, p in enumerate(pts):
        cv2.circle(salida, p, radio + 2, GRAFITO, -1)
        cv2.circle(salida, p, radio, color, -1)
        if numerar:
            cv2.putText(salida, str(i + 1), (p[0] + radio + 3, p[1] - radio), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2, cv2.LINE_AA)
    return salida


def dibujar_cancha_cenital(posiciones=(), colores=(), ancho_m: float = 9.0, largo_m: float = 18.0,
                           px_por_m: int = 30, margen_m: float = 2.0, balon=None) -> np.ndarray:
    """Dibuja una cancha de voleibol vista desde arriba con jugadores y balón.

    Orientación de la cámara trasera: ``x`` (0–9 m) de izquierda a derecha y
    ``y`` (0–18 m) de la línea de fondo rival (arriba) a la propia (abajo); la red
    está en ``y = 9``. ``posiciones`` son coordenadas ``(x, y)`` en metros.
    """
    W = int((ancho_m + 2 * margen_m) * px_por_m)
    H = int((largo_m + 2 * margen_m) * px_por_m)
    lienzo = np.full((H, W, 3), (120, 85, 30), np.uint8)  # zona libre azul

    def px(x, y):
        return int(round((x + margen_m) * px_por_m)), int(round((y + margen_m) * px_por_m))

    cv2.rectangle(lienzo, px(0, 0), px(ancho_m, largo_m), (60, 120, 220), -1)  # cancha naranja
    blanco = (245, 245, 245)
    cv2.rectangle(lienzo, px(0, 0), px(ancho_m, largo_m), blanco, 2)
    cv2.line(lienzo, px(-0.6, largo_m / 2), px(ancho_m + 0.6, largo_m / 2), (40, 40, 40), 4)  # red
    cv2.line(lienzo, px(0, largo_m / 2), px(ancho_m, largo_m / 2), blanco, 2)
    for y in (largo_m / 2 - 3, largo_m / 2 + 3):  # líneas de ataque
        cv2.line(lienzo, px(0, y), px(ancho_m, y), blanco, 2)
    cv2.putText(lienzo, "rival", px(0.2, -0.5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blanco, 1, cv2.LINE_AA)
    cv2.putText(lienzo, "propio (camara)", px(0.2, largo_m + 1.2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blanco, 1,
                cv2.LINE_AA)
    for i, (x, y) in enumerate(posiciones):
        color = colores[i] if i < len(colores) else NARANJA
        cv2.circle(lienzo, px(x, y), 9, GRAFITO, -1)
        cv2.circle(lienzo, px(x, y), 7, color, -1)
    if balon is not None:
        cv2.circle(lienzo, px(*balon), 6, GRAFITO, -1)
        cv2.circle(lienzo, px(*balon), 4, (0, 230, 255), -1)
    return lienzo


def _estilo_ejes(ax):
    ax.set_facecolor(HEX["panel"])
    for s in ax.spines.values():
        s.set_color("#5A6478")
    ax.tick_params(colors=HEX["neutro"], labelsize=8)
    ax.xaxis.label.set_color(HEX["neutro"])
    ax.yaxis.label.set_color(HEX["neutro"])
    ax.title.set_color(HEX["neutro"])


def figura(ancho: float = 6, alto: float = 3):
    """Crea una figura Matplotlib con el estilo oscuro de la UI."""
    fig, ax = plt.subplots(figsize=(ancho, alto), dpi=110)
    fig.patch.set_facecolor(HEX["grafito"])
    _estilo_ejes(ax)
    return fig, ax


def grafica_histograma(hist: dict[str, np.ndarray], titulo: str = "", colores: dict | None = None):
    """Grafica un histograma por canal ``{"R": arr, "G": arr, ...}``."""
    por_defecto = {"R": "#FF4C4C", "G": "#00C896", "B": "#4C8DFF", "H": "#FF6B35",
                   "S": "#C77DFF", "V": "#F5F5F7", "Gris": "#F5F5F7"}
    colores = colores or por_defecto
    fig, ax = figura()
    for canal, valores in hist.items():
        ax.plot(valores, color=colores.get(canal, HEX["naranja"]), label=canal, lw=1.2)
        ax.fill_between(np.arange(len(valores)), valores, color=colores.get(canal, HEX["naranja"]), alpha=0.12)
    ax.set_title(titulo)
    ax.set_xlabel("Intensidad")
    ax.set_ylabel("Frecuencia")
    leg = ax.legend(fontsize=8, facecolor=HEX["panel"], edgecolor="#5A6478")
    for t in leg.get_texts():
        t.set_color(HEX["neutro"])
    fig.tight_layout()
    return fig


def grafica_curva(x, y, titulo="", xlabel="", ylabel="", marcas: dict | None = None):
    """Gráfica de línea simple con líneas verticales opcionales ``{"etiqueta": x}``."""
    fig, ax = figura()
    ax.plot(x, y, color=HEX["naranja"], lw=1.6)
    for i, (etq, xv) in enumerate((marcas or {}).items()):
        ax.axvline(xv, color=[HEX["verde"], HEX["rojo"], HEX["neutro"]][i % 3], ls="--", lw=1.2, label=etq)
    if marcas:
        leg = ax.legend(fontsize=8, facecolor=HEX["panel"], edgecolor="#5A6478")
        for t in leg.get_texts():
            t.set_color(HEX["neutro"])
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def grafica_barras(etiquetas, valores, titulo="", colores=None, ylabel=""):
    """Gráfica de barras (p. ej. tiempos de inferencia)."""
    fig, ax = figura(5, 2.8)
    ax.bar(etiquetas, valores, color=colores or HEX["naranja"])
    for i, v in enumerate(valores):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", color=HEX["neutro"], fontsize=8)
    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig
