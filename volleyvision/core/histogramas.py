r"""Histogramas: análisis, ecualización y comparación.

Histograma
----------
Para una imagen de :math:`N` píxeles con niveles :math:`r_k \in \{0,\dots,L-1\}`,
el histograma normalizado es la probabilidad empírica de cada nivel:

.. math::

    p(r_k) = \frac{n_k}{N}

Ecualización
------------
Se busca que los niveles queden distribuidos de forma uniforme usando como
transformación la función de distribución acumulada (CDF):

.. math::

    s_k = T(r_k) = (L - 1) \sum_{j=0}^{k} p(r_j)

En color se ecualiza solo la luminancia (canal V de HSV o L de Lab) para no
alterar los tonos.

CLAHE
-----
*Contrast Limited Adaptive Histogram Equalization*: divide la imagen en
baldosas, ecualiza cada una con su propio histograma **recortado** en
``clip_limit`` (el exceso se redistribuye, limitando la amplificación del
ruido) e interpola bilinealmente entre baldosas. Corrige la iluminación
desigual de un coliseo mejor que la ecualización global.

Comparación de histogramas
--------------------------
Para distinguir los uniformes se usa el histograma 2D de tono y saturación (H-S),
que es poco sensible al brillo. Métricas de ``cv2.compareHist``:

* Correlación: :math:`d = \frac{\sum (H_1 - \bar H_1)(H_2 - \bar H_2)}{\sqrt{\sum (H_1-\bar H_1)^2 \sum (H_2-\bar H_2)^2}}` (1 = idénticos).
* Chi-cuadrado: :math:`d = \sum \frac{(H_1 - H_2)^2}{H_1}` (0 = idénticos).
* Intersección: :math:`d = \sum \min(H_1, H_2)` (mayor = más parecidos).
* Bhattacharyya: :math:`d = \sqrt{1 - \frac{1}{\sqrt{\bar H_1 \bar H_2 N^2}} \sum \sqrt{H_1 H_2}}` (0 = idénticos).
"""

from __future__ import annotations

import cv2
import numpy as np

METODOS_COMPARACION = {
    "Correlación": (cv2.HISTCMP_CORREL, True),       # (constante, mayor = más parecido)
    "Chi-cuadrado": (cv2.HISTCMP_CHISQR, False),
    "Intersección": (cv2.HISTCMP_INTERSECT, True),
    "Bhattacharyya": (cv2.HISTCMP_BHATTACHARYYA, False),
}


def calcular_histograma(img: np.ndarray, espacio: str = "RGB", mascara: np.ndarray | None = None,
                        normalizar: bool = False) -> dict[str, np.ndarray]:
    """Histograma por canal.

    Args:
        espacio: ``"RGB"``, ``"HSV"`` o ``"Gris"``. En HSV de OpenCV, H va de 0 a 179.
        mascara: si se indica, solo cuenta los píxeles donde la máscara es > 0.
        normalizar: si es ``True`` devuelve probabilidades (suman 1).
    """
    m = None if mascara is None else (mascara > 0).astype(np.uint8)
    if espacio == "Gris" or img.ndim == 2:
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canales = {"Gris": (g, 256)}
    elif espacio == "HSV":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        canales = {"H": (hsv[..., 0], 180), "S": (hsv[..., 1], 256), "V": (hsv[..., 2], 256)}
    else:
        canales = {"R": (img[..., 2], 256), "G": (img[..., 1], 256), "B": (img[..., 0], 256)}
    salida = {}
    for nombre, (canal, bins) in canales.items():
        h = cv2.calcHist([np.ascontiguousarray(canal)], [0], m, [bins], [0, bins]).ravel()
        salida[nombre] = h / max(h.sum(), 1) if normalizar else h
    return salida


def histograma_manual(gris: np.ndarray) -> np.ndarray:
    """Conteo de niveles con NumPy (equivalente a ``cv2.calcHist`` en 256 bins)."""
    return np.bincount(gris.ravel(), minlength=256).astype(np.float64)


def ecualizar_manual(gris: np.ndarray) -> np.ndarray:
    """Ecualización implementada a mano con la CDF (misma fórmula que ``cv2.equalizeHist``).

    OpenCV usa :math:`s_k = \\mathrm{round}\\big(255 \\cdot (cdf_k - cdf_{min}) / (N - cdf_{min})\\big)`
    para que el nivel más oscuro presente quede en 0.
    """
    hist = histograma_manual(gris)
    cdf = np.cumsum(hist)
    cdf_min = cdf[np.nonzero(cdf)[0][0]]
    n = gris.size
    if n == cdf_min:
        return gris.copy()
    lut = np.clip(np.round((cdf - cdf_min) * 255.0 / (n - cdf_min)), 0, 255).astype(np.uint8)
    return lut[gris]


def ecualizar_global(img: np.ndarray) -> np.ndarray:
    """Ecualización global. En color se ecualiza el canal V de HSV."""
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def aplicar_clahe(img: np.ndarray, clip_limit: float = 2.0, baldosas: int = 8) -> np.ndarray:
    """CLAHE sobre la luminancia L del espacio Lab (o directamente en grises)."""
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(baldosas), int(baldosas)))
    if img.ndim == 2:
        return clahe.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def contraste_rms(img: np.ndarray) -> float:
    """Desviación estándar de la luminancia: medida simple de contraste."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(g.std())


def uniformidad_iluminacion(img: np.ndarray, bloques: int = 4) -> float:
    """Desviación estándar del brillo medio entre bloques (menor = iluminación más pareja)."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    medias = [g[i * h // bloques:(i + 1) * h // bloques, j * w // bloques:(j + 1) * w // bloques].mean()
              for i in range(bloques) for j in range(bloques)]
    return float(np.std(medias))


# --------------------------------------------------------------------------
# Comparación de uniformes
# --------------------------------------------------------------------------
def histograma_hs(img: np.ndarray, mascara: np.ndarray | None = None, bins=(30, 32),
                  s_min: int = 60, v_min: int = 40) -> np.ndarray:
    """Histograma 2D H-S normalizado de una región (descarta píxeles grises u oscuros).

    Los píxeles con poca saturación (blancos, grises, negros) tienen un tono
    inestable, por eso se excluyen con ``s_min`` y ``v_min``.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] >= s_min) & (hsv[..., 2] >= v_min)).astype(np.uint8)
    if mascara is not None:
        m &= (mascara > 0).astype(np.uint8)
    h = cv2.calcHist([hsv], [0, 1], m, list(bins), [0, 180, 0, 256])
    total = h.sum()
    return (h / total if total > 0 else h).astype(np.float32)


def recorte_camiseta(caja) -> tuple[int, int, int, int]:
    """Zona del torso dentro de la caja de una persona (≈ 20 %–50 % de la altura),
    que es donde está el color del uniforme."""
    x1, y1, x2, y2 = caja
    h, w = y2 - y1, x2 - x1
    return int(x1 + 0.2 * w), int(y1 + 0.2 * h), int(x2 - 0.2 * w), int(y1 + 0.5 * h)


def histogramas_de_cajas(img: np.ndarray, cajas, solo_torso: bool = True) -> list[np.ndarray]:
    """Histograma H-S de cada caja (por defecto solo del torso)."""
    hs = []
    H, W = img.shape[:2]
    for caja in cajas:
        x1, y1, x2, y2 = recorte_camiseta(caja) if solo_torso else [int(v) for v in caja]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, max(x2, x1 + 1)), min(H, max(y2, y1 + 1))
        hs.append(histograma_hs(img[y1:y2, x1:x2]))
    return hs


def comparar_histogramas(h1: np.ndarray, h2: np.ndarray, metodo: str = "Bhattacharyya") -> float:
    """Distancia/similitud entre dos histogramas con ``cv2.compareHist``."""
    return float(cv2.compareHist(h1, h2, METODOS_COMPARACION[metodo][0]))


def matriz_similitud(hists: list[np.ndarray], metodo: str = "Bhattacharyya") -> np.ndarray:
    """Matriz N×N con la comparación de cada par de histogramas."""
    n = len(hists)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = comparar_histogramas(hists[i], hists[j], metodo)
    return M


def asignar_equipos(hists: list[np.ndarray], metodo: str = "Bhattacharyya") -> tuple[list[int], np.ndarray]:
    """Agrupa las regiones en dos equipos comparando histogramas de color.

    1. Se toman como referencias el par de regiones **más distintas**.
    2. Cada región se asigna al equipo cuya referencia se parezca más.
    3. Las referencias se recalculan como el promedio de su grupo y se repite
       (k-medias con k = 2 usando la métrica del histograma).

    Returns:
        (equipo de cada región: 0 ó 1, matriz de similitud).
    """
    n = len(hists)
    M = matriz_similitud(hists, metodo)
    if n < 2:
        return [0] * n, M
    mayor_es_mejor = METODOS_COMPARACION[metodo][1]
    dist = -M if mayor_es_mejor else M.copy()
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    refs = [hists[i], hists[j]]
    equipos = [0] * n
    for _ in range(5):
        for k, h in enumerate(hists):
            d = [comparar_histogramas(h, r, metodo) for r in refs]
            equipos[k] = int(np.argmax(d) if mayor_es_mejor else np.argmin(d))
        for e in (0, 1):
            grupo = [hists[k] for k in range(n) if equipos[k] == e]
            if grupo:
                refs[e] = (sum(grupo) / len(grupo)).astype(np.float32)
    return equipos, M


def color_dominante(img: np.ndarray, caja) -> tuple[int, int, int]:
    """Color BGR medio del torso (útil para pintar el equipo en el mapa)."""
    x1, y1, x2, y2 = recorte_camiseta(caja)
    H, W = img.shape[:2]
    reg = img[max(0, y1):min(H, max(y2, y1 + 1)), max(0, x1):min(W, max(x2, x1 + 1))]
    if reg.size == 0:
        return (128, 128, 128)
    return tuple(int(v) for v in np.median(reg.reshape(-1, 3), axis=0))
