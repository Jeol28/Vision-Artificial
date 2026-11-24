r"""Segmentación clásica: contornos, watershed y k-means.

Segmentar es dividir la imagen en regiones con significado (jugadores, balón,
cancha). Aquí están los métodos clásicos. Los basados en transformers
(SegFormer, SAM) están en :mod:`volleyvision.core.transformers_models` y se
comparan con estos en la UI.

Contornos
---------
``cv2.findContours`` (algoritmo de Suzuki–Abe) recorre la frontera de cada
componente conexa de una máscara binaria y la devuelve como una lista de puntos.

Watershed con transformada de distancia
---------------------------------------
Cuando dos jugadores se tocan, su máscara forma una sola mancha. La
**transformada de distancia** :math:`D(p) = \min_{q \notin A} \lVert p - q \rVert`
tiene un máximo en el centro de cada cuerpo; esos máximos son los *marcadores*.
Watershed trata la imagen como un relieve topográfico y "inunda" desde los
marcadores; donde se encuentran dos cuencas se trazan las fronteras (−1).

K-means en color
----------------
Agrupa los píxeles en :math:`k` colores minimizando

.. math::

    J = \sum_{j=1}^{k} \sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert^2

alternando asignación al centro más cercano y recálculo de centros (Lloyd).
En Lab, la distancia euclidiana se parece más a la diferencia de color
percibida que en RGB. En una toma de voleibol los grupos más grandes son el
fondo (cancha, zona libre, gradas) y los pequeños son los jugadores y el balón.
"""

from __future__ import annotations

import cv2
import numpy as np

from volleyvision.core import morfologia as mo

ESPACIOS_KMEANS = {"Lab": (cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2BGR), "HSV": (cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR),
                   "BGR": (None, None)}


# --------------------------------------------------------------------------
# Contornos
# --------------------------------------------------------------------------
def encontrar_contornos(mascara: np.ndarray, area_min: float = 0, solo_externos: bool = True) -> list[np.ndarray]:
    """Contornos de una máscara binaria con área ≥ ``area_min``, de mayor a menor."""
    modo = cv2.RETR_EXTERNAL if solo_externos else cv2.RETR_LIST
    contornos, _ = cv2.findContours((mascara > 0).astype(np.uint8), modo, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [c for c in contornos if cv2.contourArea(c) >= area_min]
    return sorted(contornos, key=cv2.contourArea, reverse=True)


def dibujar_contornos(img: np.ndarray, contornos, color=(53, 107, 255), grosor: int = 2,
                      con_caja: bool = False) -> np.ndarray:
    salida = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(salida, contornos, -1, color, grosor)
    if con_caja:
        for c in contornos:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(salida, (x, y), (x + w, y + h), (150, 200, 0), 1)
    return salida


def etiquetas_desde_contornos(forma, contornos) -> np.ndarray:
    """Imagen de etiquetas (1..N) rellenando cada contorno."""
    etiquetas = np.zeros(forma[:2], np.int32)
    for i, c in enumerate(contornos, start=1):
        cv2.drawContours(etiquetas, [c], -1, i, -1)
    return etiquetas


# --------------------------------------------------------------------------
# Watershed
# --------------------------------------------------------------------------
def watershed_distancia(img: np.ndarray, mascara: np.ndarray, frac_umbral: float = 0.5,
                        tam_fondo: int = 3) -> dict[str, np.ndarray]:
    """Separa objetos que se tocan con watershed sobre la transformada de distancia.

    1. ``fondo_seguro`` = dilatación de la máscara.
    2. ``distancia`` = transformada de distancia euclidiana de la máscara.
    3. ``frente_seguro`` = píxeles con distancia ≥ ``frac_umbral`` · (máximo de **su**
       componente); normalizar por componente evita perder los jugadores lejanos,
       que son más pequeños.
    4. Marcadores = componentes del frente seguro; zona desconocida = 0.
    5. ``cv2.watershed`` inunda desde los marcadores sobre el gradiente de la imagen.

    Returns:
        ``etiquetas`` (−1 = frontera, 1 = fondo, ≥2 = objetos), ``distancia`` (uint8),
        ``frente_seguro``, ``desconocido``, ``resultado`` (imagen con fronteras) y ``n_objetos``.
    """
    m = (mascara > 0).astype(np.uint8) * 255
    ee = mo.elemento_estructurante("Elipse", tam_fondo)
    fondo_seguro = cv2.dilate(m, ee, iterations=2)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    n, comp = cv2.connectedComponents(m)
    maximos = np.zeros(n, np.float32)
    np.maximum.at(maximos, comp.ravel(), dist.ravel())
    frente = ((dist >= frac_umbral * maximos[comp]) & (m > 0)).astype(np.uint8) * 255
    desconocido = cv2.subtract(fondo_seguro, frente)
    n_obj, marcadores = cv2.connectedComponents(frente)
    marcadores = marcadores + 1
    marcadores[desconocido > 0] = 0
    color = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    etiquetas = cv2.watershed(color.copy(), marcadores.astype(np.int32))
    resultado = color.copy()
    resultado[etiquetas == -1] = (53, 107, 255)
    return {"etiquetas": etiquetas, "distancia": cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            "frente_seguro": frente, "desconocido": desconocido, "resultado": resultado, "n_objetos": n_obj - 1}


# --------------------------------------------------------------------------
# K-means
# --------------------------------------------------------------------------
def kmeans_color(img: np.ndarray, k: int = 6, espacio: str = "Lab", intentos: int = 3,
                 max_lado: int = 320, semilla: int = 0) -> dict:
    """Cuantiza la imagen en ``k`` colores con ``cv2.kmeans``.

    El entrenamiento se hace sobre una versión reducida (``max_lado``) para que sea
    rápido en CPU; luego cada píxel de la imagen completa se asigna al centro más cercano.

    Returns:
        ``etiquetas`` (H×W), ``centros_bgr`` (k×3), ``cuantizada`` (imagen) y
        ``proporciones`` (fracción del área de cada grupo).
    """
    ida, vuelta = ESPACIOS_KMEANS[espacio]
    conv = cv2.cvtColor(img, ida) if ida is not None else img
    h, w = img.shape[:2]
    esc = min(1.0, max_lado / max(h, w))
    peq = cv2.resize(conv, (max(1, int(w * esc)), max(1, int(h * esc))), interpolation=cv2.INTER_AREA)
    datos = peq.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(semilla)
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, centros = cv2.kmeans(datos, int(k), None, criterio, int(intentos), cv2.KMEANS_PP_CENTERS)
    todos = conv.reshape(-1, 3).astype(np.float32)
    d = ((todos[:, None, :] - centros[None, :, :]) ** 2).sum(-1)
    etiquetas = d.argmin(1).reshape(h, w)
    centros_u8 = np.clip(centros, 0, 255).astype(np.uint8)
    centros_bgr = cv2.cvtColor(centros_u8[None], vuelta)[0] if vuelta is not None else centros_u8
    cuantizada = centros_bgr[etiquetas]
    proporciones = np.bincount(etiquetas.ravel(), minlength=int(k)) / etiquetas.size
    return {"etiquetas": etiquetas, "centros_bgr": centros_bgr, "cuantizada": cuantizada,
            "proporciones": proporciones}


def mascara_primer_plano_kmeans(img: np.ndarray, k: int = 4, frac_fondo: float = 0.08,
                                peso_luminancia: float = 0.5) -> tuple[np.ndarray, dict]:
    """Sustracción de fondo por modelo de color.

    1. k-means en Lab encuentra los colores dominantes; los grupos con al menos
       ``frac_fondo`` del área se toman como **fondo** (piso, zona libre, gradas).
    2. Para cada píxel se calcula la distancia de color al centro de fondo más
       cercano, dando menos peso a L (``peso_luminancia``) para tolerar sombras.
    3. Otsu sobre ese mapa de distancias separa fondo y primer plano.

    Returns:
        (máscara de primer plano, dict de k-means + ``distancia`` uint8 + ``T_otsu``).
    """
    km = kmeans_color(img, k, "Lab")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    centros_lab = cv2.cvtColor(km["centros_bgr"][None], cv2.COLOR_BGR2LAB)[0].astype(np.float32)
    fondo = centros_lab[km["proporciones"] >= frac_fondo]
    if len(fondo) == 0:
        fondo = centros_lab[[int(np.argmax(km["proporciones"]))]]
    w = np.array([peso_luminancia, 1.0, 1.0], np.float32)
    d = np.sqrt((((lab[:, :, None, :] - fondo[None, None]) * w) ** 2).sum(-1)).min(-1)
    d8 = np.clip(d * 2, 0, 255).astype(np.uint8)
    T, mascara = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    km.update({"distancia": d8, "T_otsu": int(T), "es_fondo": km["proporciones"] >= frac_fondo})
    return mascara, km


def mascara_lineas_blancas(img: np.ndarray, s_max: int = 50, v_min: int = 170) -> np.ndarray:
    """Píxeles blancos (poca saturación, mucho brillo): líneas de la cancha y red."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 1] <= s_max) & (hsv[..., 2] >= v_min)).astype(np.uint8) * 255


# --------------------------------------------------------------------------
# Jugadores (pipeline clásico)
# --------------------------------------------------------------------------
def segmentar_jugadores_clasico(img: np.ndarray, k: int = 6, frac_fondo: float = 0.08,
                                tam_apertura: int = 3, tam_cierre: int = 9, area_min: int = 150,
                                relacion_min: float = 1.3, usar_watershed: bool = False,
                                quitar_lineas: bool = True,
                                mascara_excluir: np.ndarray | None = None) -> dict:
    """Detecta jugadores sin aprendizaje:
    modelo de fondo con k-means + Otsu → quitar líneas blancas → morfología →
    (watershed) → contornos.

    Se conservan las regiones "altas" (alto/ancho ≥ ``relacion_min``), como una persona de pie.

    Args:
        mascara_excluir: píxeles a descartar (p. ej. la máscara del balón).

    Returns:
        ``mascara`` limpia, ``mascara_cruda``, ``regiones`` (lista con ``caja``,
        ``contorno``, ``area``) y ``kmeans``.
    """
    cruda, km = mascara_primer_plano_kmeans(img, k, frac_fondo)
    if quitar_lineas:
        cruda[mascara_lineas_blancas(img) > 0] = 0
    if mascara_excluir is not None:
        cruda[mascara_excluir > 0] = 0
    # Apertura: quita ruido. Cierre vertical: une camiseta, pantaloneta y piernas
    # (una persona de pie es alargada en vertical). Cierre elíptico: rellena huecos.
    limpia = mo.apertura(cruda, mo.elemento_estructurante("Elipse", tam_apertura)) if tam_apertura > 1 else cruda
    ee_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2 * tam_cierre + 1))
    limpia = mo.cierre(limpia, ee_vertical)
    limpia = mo.limpiar_mascara(limpia, 0, max(3, tam_cierre // 2), "Elipse", area_min=area_min)
    if usar_watershed:
        ws = watershed_distancia(img, limpia, 0.6)
        etiquetas = ws["etiquetas"]
        ids = [i for i in np.unique(etiquetas) if i > 1]
        mascaras = [(etiquetas == i).astype(np.uint8) * 255 for i in ids]
    else:
        mascaras = [etiquetas_desde_contornos(img.shape, [c]) > 0 for c in encontrar_contornos(limpia, area_min)]
        mascaras = [m.astype(np.uint8) * 255 for m in mascaras]
    regiones = []
    for m in mascaras:
        cs = encontrar_contornos(m, area_min)
        if not cs:
            continue
        c = cs[0]
        x, y, w, h = cv2.boundingRect(c)
        if h / max(w, 1) >= relacion_min:
            regiones.append({"caja": (x, y, x + w, y + h), "contorno": c, "area": float(cv2.contourArea(c))})
    return {"mascara": limpia, "mascara_cruda": cruda, "regiones": regiones, "kmeans": km}


# --------------------------------------------------------------------------
# Métricas de comparación
# --------------------------------------------------------------------------
def iou_mascaras(a: np.ndarray, b: np.ndarray) -> float:
    """Intersección sobre unión de dos máscaras binarias."""
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def iou_cajas(a, b) -> float:
    """IoU entre dos cajas (x1, y1, x2, y2)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter) if inter > 0 else 0.0


def emparejar_cajas(pred, reales, umbral_iou: float = 0.5) -> dict:
    """Emparejamiento voraz por IoU: devuelve VP, FP, FN, precisión, exhaustividad y la
    lista ``correctas`` (bool por predicción) para colorear aciertos y errores."""
    pares = sorted(((iou_cajas(p, r), i, j) for i, p in enumerate(pred) for j, r in enumerate(reales)),
                   reverse=True)
    usadas_p, usadas_r = set(), set()
    for v, i, j in pares:
        if v < umbral_iou:
            break
        if i not in usadas_p and j not in usadas_r:
            usadas_p.add(i)
            usadas_r.add(j)
    vp = len(usadas_p)
    fp = len(pred) - vp
    fn = len(reales) - vp
    return {"VP": vp, "FP": fp, "FN": fn,
            "precision": vp / len(pred) if pred else 0.0,
            "exhaustividad": vp / len(reales) if reales else 0.0,
            "correctas": [i in usadas_p for i in range(len(pred))]}
