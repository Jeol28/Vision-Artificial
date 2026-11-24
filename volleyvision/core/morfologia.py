r"""Morfología matemática.

Opera sobre la **forma** de los objetos en una imagen binaria (o de grises)
usando un elemento estructurante :math:`B` (rectángulo, elipse o cruz).

* **Erosión**: :math:`A \ominus B = \{z \mid B_z \subseteq A\}`. En grises es el
  mínimo del vecindario. Adelgaza objetos y elimina puntos aislados.
* **Dilatación**: :math:`A \oplus B = \{z \mid (\hat B)_z \cap A \neq \emptyset\}`.
  En grises es el máximo del vecindario. Engrosa objetos y rellena huecos.
* **Apertura**: :math:`A \circ B = (A \ominus B) \oplus B`. Elimina objetos más
  pequeños que B (ruido) conservando el tamaño de los grandes.
* **Cierre**: :math:`A \bullet B = (A \oplus B) \ominus B`. Rellena huecos y une
  fragmentos cercanos (p. ej. las franjas del balón).
* **Gradiente morfológico**: :math:`(A \oplus B) - (A \ominus B)`; da el contorno.
* **Top-hat**: :math:`A - (A \circ B)`; resalta detalles claros pequeños (líneas).

Erosión y dilatación son **duales**: :math:`(A \ominus B)^c = A^c \oplus \hat B`.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

FORMAS = {"Rectángulo": cv2.MORPH_RECT, "Elipse": cv2.MORPH_ELLIPSE, "Cruz": cv2.MORPH_CROSS}
OPERACIONES = ("Erosión", "Dilatación", "Apertura", "Cierre", "Gradiente", "Top-hat", "Black-hat")


def elemento_estructurante(forma: str = "Elipse", tam: int = 5) -> np.ndarray:
    """Crea el elemento estructurante (``cv2.getStructuringElement``) de tamaño ``tam``×``tam``."""
    tam = max(1, int(tam))
    return cv2.getStructuringElement(FORMAS[forma], (tam, tam))


def erosion(img: np.ndarray, ee: np.ndarray, iteraciones: int = 1) -> np.ndarray:
    return cv2.erode(img, ee, iterations=int(iteraciones))


def dilatacion(img: np.ndarray, ee: np.ndarray, iteraciones: int = 1) -> np.ndarray:
    return cv2.dilate(img, ee, iterations=int(iteraciones))


def apertura(img: np.ndarray, ee: np.ndarray, iteraciones: int = 1) -> np.ndarray:
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, ee, iterations=int(iteraciones))


def cierre(img: np.ndarray, ee: np.ndarray, iteraciones: int = 1) -> np.ndarray:
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, ee, iterations=int(iteraciones))


def gradiente(img: np.ndarray, ee: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, ee)


def aplicar(img: np.ndarray, operacion: str, ee: np.ndarray, iteraciones: int = 1) -> np.ndarray:
    """Aplica la operación indicada por nombre (ver :data:`OPERACIONES`)."""
    ops = {
        "Erosión": lambda: erosion(img, ee, iteraciones),
        "Dilatación": lambda: dilatacion(img, ee, iteraciones),
        "Apertura": lambda: apertura(img, ee, iteraciones),
        "Cierre": lambda: cierre(img, ee, iteraciones),
        "Gradiente": lambda: gradiente(img, ee),
        "Top-hat": lambda: cv2.morphologyEx(img, cv2.MORPH_TOPHAT, ee),
        "Black-hat": lambda: cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, ee),
    }
    if operacion not in ops:
        raise ValueError(f"Operación desconocida: {operacion}")
    return ops[operacion]()


# --------------------------------------------------------------------------
# Implementaciones manuales (para entender la definición)
# --------------------------------------------------------------------------
def _vecindarios(img: np.ndarray, ee: np.ndarray, relleno: int) -> np.ndarray:
    a, b = ee.shape[0] // 2, ee.shape[1] // 2
    rellena = np.pad(img, ((a, a), (b, b)), mode="constant", constant_values=relleno)
    return sliding_window_view(rellena, ee.shape)


def erosion_manual(img: np.ndarray, ee: np.ndarray) -> np.ndarray:
    """Erosión en grises = mínimo del vecindario definido por el elemento estructurante.

    El borde se rellena con 255 (valor neutro del mínimo), igual que OpenCV por defecto.
    """
    v = _vecindarios(img, ee, 255)
    return np.where(ee.astype(bool), v, 255).min(axis=(-2, -1)).astype(img.dtype)


def dilatacion_manual(img: np.ndarray, ee: np.ndarray) -> np.ndarray:
    """Dilatación en grises = máximo del vecindario (se usa el elemento reflejado)."""
    ee_r = ee[::-1, ::-1]
    v = _vecindarios(img, ee_r, 0)
    return np.where(ee_r.astype(bool), v, 0).max(axis=(-2, -1)).astype(img.dtype)


# --------------------------------------------------------------------------
# Limpieza de máscaras
# --------------------------------------------------------------------------
def eliminar_regiones_pequenas(mascara: np.ndarray, area_min: int) -> np.ndarray:
    """Quita las componentes conexas con menos de ``area_min`` píxeles."""
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats((mascara > 0).astype(np.uint8), connectivity=8)
    conservar = np.zeros(n, bool)
    conservar[1:] = stats[1:, cv2.CC_STAT_AREA] >= area_min
    return np.where(conservar[etiquetas], 255, 0).astype(np.uint8)


def rellenar_huecos(mascara: np.ndarray) -> np.ndarray:
    """Rellena huecos internos (reconstrucción por *flood fill* desde el borde)."""
    m = (mascara > 0).astype(np.uint8) * 255
    h, w = m.shape
    relleno = m.copy()
    borde = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(relleno, borde, (0, 0), 255)
    return m | cv2.bitwise_not(relleno)


def limpiar_mascara(mascara: np.ndarray, tam_apertura: int = 3, tam_cierre: int = 7,
                    forma: str = "Elipse", area_min: int = 0, rellenar: bool = False) -> np.ndarray:
    """Limpieza estándar: apertura (quita ruido) → cierre (une fragmentos) → filtros opcionales."""
    m = (mascara > 0).astype(np.uint8) * 255
    if tam_apertura > 1:
        m = apertura(m, elemento_estructurante(forma, tam_apertura))
    if tam_cierre > 1:
        m = cierre(m, elemento_estructurante(forma, tam_cierre))
    if rellenar:
        m = rellenar_huecos(m)
    if area_min > 0:
        m = eliminar_regiones_pequenas(m, area_min)
    return m


def contar_componentes(mascara: np.ndarray) -> int:
    """Número de componentes conexas (sin contar el fondo)."""
    return cv2.connectedComponents((mascara > 0).astype(np.uint8), connectivity=8)[0] - 1
