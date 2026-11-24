r"""Umbralización y binarización.

Umbral global
-------------
.. math::

    g(x, y) = \begin{cases} 255 & f(x, y) > T \\ 0 & f(x, y) \le T \end{cases}

Método de Otsu
--------------
Elige el T que mejor separa el histograma en dos clases (fondo y objeto),
maximizando la **varianza entre clases**. Con :math:`p_i` el histograma
normalizado:

.. math::

    \omega_0(t) = \sum_{i=0}^{t} p_i, \qquad
    \mu(t) = \sum_{i=0}^{t} i\, p_i, \qquad
    \mu_T = \sum_{i=0}^{L-1} i\, p_i

.. math::

    \sigma_B^2(t) = \frac{\big(\mu_T\, \omega_0(t) - \mu(t)\big)^2}{\omega_0(t)\,\big(1 - \omega_0(t)\big)},
    \qquad T^* = \arg\max_t \sigma_B^2(t)

Maximizar :math:`\sigma_B^2` equivale a minimizar la varianza dentro de cada
clase, porque :math:`\sigma_T^2 = \sigma_W^2 + \sigma_B^2` es constante.

Umbral adaptativo
-----------------
Cada píxel usa su propio umbral calculado en un vecindario :math:`b\times b`:
:math:`T(x, y) = \text{media}_{b\times b}(x, y) - C` (o media ponderada
gaussiana). Funciona con iluminación no uniforme.

Umbral por rango de color (HSV)
-------------------------------
En HSV el tono (H) está separado del brillo (V), así que un intervalo
:math:`[H_{min}, H_{max}] \times [S_{min}, S_{max}] \times [V_{min}, V_{max}]`
aísla un color (el amarillo del balón, el naranja de la cancha) aunque cambie
la iluminación. En OpenCV H ∈ [0, 179] (grados / 2).
"""

from __future__ import annotations

import cv2
import numpy as np

# Rangos HSV de OpenCV (H 0–179, S y V 0–255): (bajo, alto)
RANGOS_HSV: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "Balón (amarillo)": ((19, 120, 110), (34, 255, 255)),
    "Balón (azul)": ((95, 120, 60), (125, 255, 255)),
    "Cancha (naranja)": ((4, 130, 50), (18, 255, 255)),
    "Zona libre (verde)": ((40, 80, 40), (85, 255, 255)),
    "Uniforme rojo": ((0, 150, 50), (4, 255, 255)),
    "Uniforme azul": ((100, 150, 50), (130, 255, 255)),
    "Líneas (blanco)": ((0, 0, 170), (179, 50, 255)),
}


def _gris(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def umbral_global(img: np.ndarray, T: int, invertir: bool = False) -> np.ndarray:
    """Binarización con umbral fijo T (implementación directa con NumPy)."""
    g = _gris(img)
    binaria = np.where(g > T, 255, 0).astype(np.uint8)
    return 255 - binaria if invertir else binaria


def otsu_manual(img: np.ndarray) -> tuple[int, np.ndarray]:
    """Método de Otsu implementado a mano.

    Returns:
        (T óptimo, curva de varianza entre clases σ²_B(t) para t = 0..255).
    """
    g = _gris(img)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    niveles = np.arange(256)
    omega = np.cumsum(p)                 # ω0(t)
    mu = np.cumsum(niveles * p)          # μ(t)
    mu_t = mu[-1]                        # μT
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / denom, 0.0)
    return int(np.argmax(sigma_b)), sigma_b


def otsu_opencv(img: np.ndarray, invertir: bool = False) -> tuple[int, np.ndarray]:
    """Otsu con ``cv2.threshold(..., THRESH_OTSU)``. Devuelve (T, imagen binaria)."""
    tipo = cv2.THRESH_BINARY_INV if invertir else cv2.THRESH_BINARY
    T, binaria = cv2.threshold(_gris(img), 0, 255, tipo + cv2.THRESH_OTSU)
    return int(T), binaria


def umbral_adaptativo(img: np.ndarray, metodo: str = "gaussiano", bloque: int = 31, C: float = 5,
                      invertir: bool = False) -> np.ndarray:
    """Umbral adaptativo por media o por media gaussiana del vecindario ``bloque``×``bloque``."""
    bloque = max(3, int(bloque) | 1)
    modo = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if metodo == "gaussiano" else cv2.ADAPTIVE_THRESH_MEAN_C
    tipo = cv2.THRESH_BINARY_INV if invertir else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(_gris(img), 255, modo, tipo, bloque, float(C))


def umbral_hsv(img: np.ndarray, bajo, alto) -> np.ndarray:
    """Máscara de los píxeles cuyo color HSV está en [bajo, alto].

    Si ``bajo[0] > alto[0]`` el rango de tono da la vuelta al círculo (p. ej. rojos
    de 170 a 10) y se unen los intervalos [bajo, 179] y [0, alto].
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bajo = np.array(bajo, np.uint8)
    alto = np.array(alto, np.uint8)
    if bajo[0] <= alto[0]:
        return cv2.inRange(hsv, bajo, alto)
    m1 = cv2.inRange(hsv, bajo, np.array([179, alto[1], alto[2]], np.uint8))
    m2 = cv2.inRange(hsv, np.array([0, bajo[1], bajo[2]], np.uint8), alto)
    return cv2.bitwise_or(m1, m2)


def mascara_por_nombre(img: np.ndarray, nombre: str) -> np.ndarray:
    """Máscara HSV usando un rango predefinido de :data:`RANGOS_HSV`."""
    bajo, alto = RANGOS_HSV[nombre]
    return umbral_hsv(img, bajo, alto)


def fraccion_blanca(mascara: np.ndarray) -> float:
    """Proporción de píxeles activos de una máscara."""
    return float((mascara > 0).mean())
