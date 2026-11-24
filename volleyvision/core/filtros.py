r"""Filtros espaciales: suavizado, detección de bordes y convolución.

Convolución 2D
--------------
Un filtro lineal reemplaza cada píxel por una combinación ponderada de su
vecindario, definida por un *kernel* :math:`w` de tamaño :math:`(2a+1)\times(2b+1)`:

.. math::

    g(x, y) = (w * f)(x, y) = \sum_{s=-a}^{a} \sum_{t=-b}^{b} w(s, t)\, f(x - s, y - t)

La **correlación** usa :math:`f(x + s, y + t)` (no voltea el kernel). OpenCV
(``cv2.filter2D``) calcula correlación; para kernels simétricos ambas coinciden.

Suavizado
---------
* **Promedio**: kernel constante :math:`1/k^2`. Reduce ruido pero difumina bordes.
* **Gaussiano**: :math:`w(s,t) \propto e^{-(s^2+t^2)/2\sigma^2}`. Pondera más el
  centro; es separable y no introduce artefactos en anillo.
* **Mediana** (no lineal): toma la mediana del vecindario; ideal contra ruido
  sal y pimienta y preserva bordes.
* **Bilateral** (no lineal): pondera por distancia espacial **y** por similitud
  de intensidad, :math:`w = e^{-\lVert p-q\rVert^2/2\sigma_s^2}\,e^{-|f_p-f_q|^2/2\sigma_r^2}`,
  así suaviza el piso sin borrar las líneas de la cancha.

Bordes
------
* **Sobel**: aproxima :math:`\partial f/\partial x` y :math:`\partial f/\partial y`;
  magnitud :math:`\lVert\nabla f\rVert = \sqrt{G_x^2 + G_y^2}`.
* **Laplaciano**: :math:`\nabla^2 f = \partial^2 f/\partial x^2 + \partial^2 f/\partial y^2`;
  responde a cambios de segundo orden (cruces por cero).
* **Canny**: suavizado gaussiano → gradiente → supresión de no máximos →
  umbral con histéresis (:math:`T_{bajo}, T_{alto}`).
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SUAVIZADOS = ("Promedio", "Gaussiano", "Mediana", "Bilateral")


def _impar(k: int) -> int:
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1


def _gris(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------
# Suavizado
# --------------------------------------------------------------------------
def filtro_promedio(img: np.ndarray, k: int = 5) -> np.ndarray:
    """Filtro de caja (media aritmética) de tamaño k×k."""
    k = _impar(k)
    return cv2.blur(img, (k, k))


def filtro_gaussiano(img: np.ndarray, k: int = 5, sigma: float = 0.0) -> np.ndarray:
    """Filtro gaussiano; con ``sigma = 0`` OpenCV la calcula a partir de k."""
    k = _impar(k)
    return cv2.GaussianBlur(img, (k, k), sigma)


def filtro_mediana(img: np.ndarray, k: int = 5) -> np.ndarray:
    """Filtro de mediana (no lineal) de tamaño k×k."""
    return cv2.medianBlur(img, _impar(k))


def filtro_bilateral(img: np.ndarray, d: int = 9, sigma_color: float = 75,
                     sigma_espacio: float = 75) -> np.ndarray:
    """Filtro bilateral: suaviza preservando bordes."""
    return cv2.bilateralFilter(img, int(d), float(sigma_color), float(sigma_espacio))


def suavizar(img: np.ndarray, metodo: str, k: int = 5, sigma: float = 0.0,
             sigma_color: float = 75, sigma_espacio: float = 75) -> np.ndarray:
    """Despacha al filtro de suavizado elegido (``SUAVIZADOS``)."""
    if metodo == "Promedio":
        return filtro_promedio(img, k)
    if metodo == "Gaussiano":
        return filtro_gaussiano(img, k, sigma)
    if metodo == "Mediana":
        return filtro_mediana(img, k)
    if metodo == "Bilateral":
        return filtro_bilateral(img, k, sigma_color, sigma_espacio)
    raise ValueError(f"Método desconocido: {metodo}")


def agregar_ruido(img: np.ndarray, tipo: str = "gaussiano", cantidad: float = 0.05,
                  semilla: int = 0) -> np.ndarray:
    """Agrega ruido ``gaussiano`` (σ = cantidad·255) o ``sal y pimienta`` (fracción de píxeles)."""
    rng = np.random.default_rng(semilla)
    if tipo == "gaussiano":
        ruido = rng.normal(0, cantidad * 255, img.shape)
        return np.clip(img.astype(np.float32) + ruido, 0, 255).astype(np.uint8)
    salida = img.copy()
    m = rng.random(img.shape[:2])
    salida[m < cantidad / 2] = 0
    salida[m > 1 - cantidad / 2] = 255
    return salida


# --------------------------------------------------------------------------
# Bordes
# --------------------------------------------------------------------------
def sobel(img: np.ndarray, k: int = 3) -> dict[str, np.ndarray]:
    """Gradientes de Sobel.

    Returns:
        ``gx``, ``gy`` (float), ``magnitud`` (uint8 normalizada) y ``angulo`` (grados).
    """
    g = _gris(img).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=_impar(k))
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=_impar(k))
    mag = cv2.magnitude(gx, gy)
    mag8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    return {"gx": gx, "gy": gy, "magnitud": mag8, "angulo": ang}


def laplaciano(img: np.ndarray, k: int = 3, suavizar_antes: bool = True) -> np.ndarray:
    """Laplaciano en valor absoluto (uint8). Se suele suavizar antes (LoG) por su sensibilidad al ruido."""
    g = _gris(img)
    if suavizar_antes:
        g = cv2.GaussianBlur(g, (3, 3), 0)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=_impar(k))
    return cv2.convertScaleAbs(lap)


def canny(img: np.ndarray, umbral_bajo: int = 50, umbral_alto: int = 150,
          apertura: int = 3, suavizar_antes: int = 5) -> np.ndarray:
    """Detector de Canny con suavizado gaussiano previo opcional (tamaño del kernel, 0 = no)."""
    g = _gris(img)
    if suavizar_antes and suavizar_antes > 1:
        g = cv2.GaussianBlur(g, (_impar(suavizar_antes),) * 2, 0)
    return cv2.Canny(g, int(umbral_bajo), int(umbral_alto), apertureSize=_impar(apertura))


# --------------------------------------------------------------------------
# Convolución manual
# --------------------------------------------------------------------------
def convolucion_manual(img: np.ndarray, kernel: np.ndarray, voltear: bool = True) -> np.ndarray:
    """Convolución 2D implementada con NumPy (sin OpenCV).

    1. Se rellena la imagen reflejando el borde (``reflect`` de NumPy equivale a
       ``BORDER_REFLECT_101`` de OpenCV), para que la salida tenga el mismo tamaño.
    2. ``sliding_window_view`` crea, sin copiar memoria, una vista con el
       vecindario de cada píxel: forma (H, W, kh, kw).
    3. Cada salida es la suma del producto elemento a elemento entre el
       vecindario y el kernel (``einsum``).

    Args:
        img: imagen de 1 o 3 canales.
        kernel: matriz kh×kw (dimensiones impares).
        voltear: ``True`` = convolución verdadera (kernel rotado 180°);
            ``False`` = correlación (lo que hace ``cv2.filter2D``).

    Returns:
        Resultado en ``float64`` (sin recortar), del mismo tamaño que la entrada.
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("El kernel debe ser 2D y de dimensiones impares.")
    if voltear:
        kernel = kernel[::-1, ::-1]
    if img.ndim == 3:
        return np.stack([convolucion_manual(img[..., c], kernel, voltear=False) for c in range(img.shape[2])],
                        axis=-1)
    a, b = kernel.shape[0] // 2, kernel.shape[1] // 2
    rellena = np.pad(img.astype(np.float64), ((a, a), (b, b)), mode="reflect")
    ventanas = sliding_window_view(rellena, kernel.shape)
    return np.einsum("ijkl,kl->ij", ventanas, kernel)


def convolucion_opencv(img: np.ndarray, kernel: np.ndarray, voltear: bool = True) -> np.ndarray:
    """Equivalente con ``cv2.filter2D`` (que correlaciona): se voltea el kernel para convolucionar."""
    kernel = np.asarray(kernel, dtype=np.float64)
    if voltear:
        kernel = cv2.flip(kernel, -1)
    return cv2.filter2D(img.astype(np.float64), cv2.CV_64F, kernel, borderType=cv2.BORDER_REFLECT_101)


def comparar_convoluciones(img: np.ndarray, kernel: np.ndarray) -> dict:
    """Ejecuta ambas implementaciones y reporta tiempos y diferencia máxima."""
    t0 = time.perf_counter()
    manual = convolucion_manual(img, kernel)
    t1 = time.perf_counter()
    ocv = convolucion_opencv(img, kernel)
    t2 = time.perf_counter()
    return {"manual": manual, "opencv": ocv, "tiempo_manual_ms": (t1 - t0) * 1e3,
            "tiempo_opencv_ms": (t2 - t1) * 1e3, "diferencia_max": float(np.abs(manual - ocv).max())}


def a_visualizable(resultado: np.ndarray, modo: str = "recortar") -> np.ndarray:
    """Convierte el resultado float de una convolución a uint8.

    ``recortar`` satura a [0, 255] (útil para suavizados/realce); ``absoluto``
    toma |x| y normaliza (útil para kernels de bordes con valores negativos).
    """
    if modo == "absoluto":
        return cv2.normalize(np.abs(resultado), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.clip(resultado, 0, 255).astype(np.uint8)


KERNELS_PREDEFINIDOS: dict[str, list[list[float]]] = {
    "Enfoque (sharpen)": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    "Sobel X": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    "Sobel Y": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    "Laplaciano": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
    "Repujado (emboss)": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
    "Promedio 3×3": [[1 / 9] * 3] * 3,
    "Línea horizontal": [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
}


def kernel_desde_texto(texto: str) -> np.ndarray:
    """Interpreta un kernel escrito por el usuario: filas separadas por saltos de
    línea o ``;`` y valores por espacios o comas. Admite fracciones como ``1/9``."""
    filas = [f for f in texto.replace(";", "\n").strip().splitlines() if f.strip()]

    def valor(v: str) -> float:
        if "/" in v:
            n, d = v.split("/")
            return float(n) / float(d)
        return float(v)

    matriz = [[valor(v) for v in f.replace(",", " ").split()] for f in filas]
    if not matriz or len({len(f) for f in matriz}) != 1:
        raise ValueError("Todas las filas del kernel deben tener el mismo número de valores.")
    k = np.array(matriz, dtype=np.float64)
    if k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise ValueError("El kernel debe tener dimensiones impares (3×3, 5×5, ...).")
    return k


def kernel_a_texto(kernel) -> str:
    """Representación del kernel para mostrar en un cuadro de texto."""
    return "\n".join(" ".join(f"{v:g}" if abs(v - round(v)) < 1e-9 else f"{v:.4f}" for v in fila)
                     for fila in np.asarray(kernel))


# --------------------------------------------------------------------------
# Aplicación: resaltar las líneas de la cancha
# --------------------------------------------------------------------------
def resaltar_lineas_cancha(img: np.ndarray, d_bilateral: int = 9, umbral_bajo: int = 50,
                           umbral_alto: int = 150, votos_hough: int = 80,
                           long_min: int = 60, hueco_max: int = 10) -> dict[str, np.ndarray]:
    """Resalta las líneas blancas de la cancha.

    1. Filtro bilateral: alisa la textura del piso sin difuminar las líneas.
    2. Canny: bordes finos de las líneas.
    3. Transformada de Hough probabilística: segmentos rectos largos.

    Returns:
        ``suavizada``, ``bordes`` y ``lineas`` (imagen con los segmentos dibujados), y
        ``segmentos`` (N×4 con x1, y1, x2, y2).
    """
    suave = filtro_bilateral(img, d_bilateral, 60, 60)
    bordes = canny(suave, umbral_bajo, umbral_alto, suavizar_antes=0)
    segs = cv2.HoughLinesP(bordes, 1, np.pi / 180, int(votos_hough), minLineLength=int(long_min),
                           maxLineGap=int(hueco_max))
    segs = np.zeros((0, 4), int) if segs is None else segs.reshape(-1, 4)
    salida = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in segs:
        cv2.line(salida, (int(x1), int(y1)), (int(x2), int(y2)), (53, 107, 255), 2, cv2.LINE_AA)
    return {"suavizada": suave, "bordes": bordes, "lineas": salida, "segmentos": segs}
