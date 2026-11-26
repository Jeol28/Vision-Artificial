r"""Caracterización de regiones: descriptores de forma y clasificación simple.

Momentos
--------
Para una región :math:`R`, los momentos geométricos son
:math:`m_{pq} = \sum_{(x,y)\in R} x^p y^q`. De ellos salen:

* **Área**: :math:`A = m_{00}`.
* **Centroide**: :math:`(\bar x, \bar y) = (m_{10}/m_{00},\ m_{01}/m_{00})`.
* **Momentos centrales** :math:`\mu_{pq} = \sum (x-\bar x)^p (y-\bar y)^q`
  (invariantes a traslación) y normalizados
  :math:`\eta_{pq} = \mu_{pq} / \mu_{00}^{1 + (p+q)/2}` (también a escala).
* **Momentos de Hu**: 7 combinaciones de :math:`\eta_{pq}` invariantes a
  traslación, escala y rotación (:math:`\phi_7` cambia de signo con un reflejo).
  Se muestran como :math:`-\mathrm{sign}(\phi)\log_{10}|\phi|`.

Descriptores de forma
---------------------
* **Perímetro** :math:`P`: longitud del contorno.
* **Circularidad**: :math:`C = 4\pi A / P^2` (1 para un círculo perfecto).
* **Excentricidad**: con los valores propios :math:`\lambda_1 \ge \lambda_2` de la
  matriz de covarianza :math:`\begin{bmatrix}\mu_{20} & \mu_{11}\\ \mu_{11} & \mu_{02}\end{bmatrix}`,
  :math:`e = \sqrt{1 - \lambda_2/\lambda_1}` (0 = círculo, → 1 = segmento).
* **Solidez**: :math:`A / A_{\text{envolvente convexa}}` (1 = convexa).
* **Extensión**: :math:`A / A_{\text{caja}}`.

Clasificación "¿es balón?"
--------------------------
Regla interpretable: una región es balón si es casi circular (:math:`C`
alta, :math:`e` baja, solidez alta) **y** la mayoría de sus píxeles tienen el
color del balón. Los jugadores proyectan su **punto de apoyo** (centro inferior
de la caja, los pies sobre el piso) a la vista cenital con la homografía de la
cancha, porque la homografía solo es válida para puntos del plano del piso.

Solo el equipo propio
---------------------
La cámara está detrás del equipo propio: el rival queda lejos, pequeño y tapado
por la red y por los propios jugadores, así que no se analiza. Para decidir
quién es del equipo propio (:func:`seleccionar_equipo_propio`):

* **Con la homografía** (esquinas marcadas): los que pisan la media cancha
  propia, :math:`y \ge 9` m (la red está en :math:`y = 9`), con poco margen a
  los lados, hasta 5,5 m detrás de la línea de fondo (ahí está quien saca) y sin
  margen lateral detrás de ella (así quedan fuera los jueces de línea de las esquinas).
* **Sin ella**: de los dos grupos de camisetas (histogramas, tema 3), el que tiene
  los pies más abajo en la imagen, es decir, más cerca de la cámara.
"""

from __future__ import annotations

import cv2
import numpy as np

from volleyvision.core import calibracion as cal
from volleyvision.core import morfologia as mo
from volleyvision.core import segmentacion as sg
from volleyvision.core import umbralizacion as um


def momentos_hu_log(momentos: dict) -> np.ndarray:
    """Los 7 momentos de Hu en escala logarítmica con signo."""
    hu = cv2.HuMoments(momentos).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(hu != 0, -np.sign(hu) * np.log10(np.abs(hu)), 0.0)


def excentricidad_y_orientacion(m: dict) -> tuple[float, float]:
    """Excentricidad (de la elipse con los mismos segundos momentos) y orientación en grados."""
    if m["m00"] == 0:
        return 0.0, 0.0
    mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    comun = np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    l1 = (mu20 + mu02 + comun) / 2
    l2 = (mu20 + mu02 - comun) / 2
    exc = float(np.sqrt(max(0.0, 1 - l2 / l1))) if l1 > 0 else 0.0
    ang = float(np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02)))
    return exc, ang


def describir_region(contorno: np.ndarray) -> dict:
    """Calcula todos los descriptores de una región a partir de su contorno."""
    m = cv2.moments(contorno)
    area = float(m["m00"]) if m["m00"] > 0 else float(cv2.contourArea(contorno))
    perimetro = float(cv2.arcLength(contorno, True))
    x, y, w, h = cv2.boundingRect(contorno)
    if m["m00"] > 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    else:
        cx, cy = x + w / 2, y + h / 2
    area_hull = float(cv2.contourArea(cv2.convexHull(contorno)))
    exc, orient = excentricidad_y_orientacion(m)
    return {
        "area": area,
        "perimetro": perimetro,
        "centroide": (float(cx), float(cy)),
        "caja": (x, y, x + w, y + h),
        "circularidad": float(4 * np.pi * area / perimetro ** 2) if perimetro > 0 else 0.0,
        "excentricidad": exc,
        "orientacion": orient,
        "solidez": area / area_hull if area_hull > 0 else 0.0,
        "extension": area / (w * h) if w * h > 0 else 0.0,
        "relacion_aspecto": h / w if w > 0 else 0.0,
        "diametro_equivalente": float(np.sqrt(4 * area / np.pi)),
        "hu": momentos_hu_log(m),
        "contorno": contorno,
    }


def describir_mascara(mascara: np.ndarray, area_min: float = 20) -> list[dict]:
    """Descriptores de cada componente (contorno externo) de una máscara."""
    return [describir_region(c) for c in sg.encontrar_contornos(mascara, area_min)]


def tabla_descriptores(regiones: list[dict], decimales: int = 3) -> list[dict]:
    """Filas planas para mostrar en una tabla (sin el contorno)."""
    filas = []
    for i, r in enumerate(regiones, start=1):
        fila = {"id": i}
        for k, v in r.items():
            if k in ("contorno", "hu"):
                continue
            if k == "centroide":
                fila["cx"], fila["cy"] = round(v[0], 1), round(v[1], 1)
            elif k == "caja":
                fila["caja"] = str(tuple(int(a) for a in v))
            elif isinstance(v, (float, np.floating)):
                fila[k] = round(float(v), decimales)
            elif isinstance(v, (bool, np.bool_, int, str)):
                fila[k] = v
        for j, h in enumerate(r.get("hu", [])[:7], start=1):
            fila[f"Hu{j}"] = round(float(h), decimales)
        filas.append(fila)
    return filas


# --------------------------------------------------------------------------
# Clasificación del balón
# --------------------------------------------------------------------------
def fraccion_color(img: np.ndarray, contorno: np.ndarray, rangos: list[str]) -> float:
    """Fracción de píxeles dentro del contorno cuyo color cae en alguno de los rangos HSV."""
    mascara = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mascara, [contorno], -1, 255, -1)
    total = int((mascara > 0).sum())
    if total == 0:
        return 0.0
    color = np.zeros_like(mascara)
    for r in rangos:
        color |= um.mascara_por_nombre(img, r)
    return float(((color > 0) & (mascara > 0)).sum() / total)


def es_balon(desc: dict, frac_color: float, circ_min: float = 0.78, exc_max: float = 0.45,
             solidez_min: float = 0.85, frac_color_min: float = 0.4,
             area_min: float = 15, area_max: float = 20000) -> tuple[bool, float, list[str]]:
    """Clasificador por reglas: ¿la región es un balón?

    Returns:
        (decisión, puntaje 0–1 = fracción de reglas cumplidas, motivos de rechazo).
    """
    reglas = {
        f"circularidad {desc['circularidad']:.2f} < {circ_min}": desc["circularidad"] >= circ_min,
        f"excentricidad {desc['excentricidad']:.2f} > {exc_max}": desc["excentricidad"] <= exc_max,
        f"solidez {desc['solidez']:.2f} < {solidez_min}": desc["solidez"] >= solidez_min,
        f"color del balón {frac_color:.0%} < {frac_color_min:.0%}": frac_color >= frac_color_min,
        f"área {desc['area']:.0f} fuera de [{area_min}, {area_max}]": area_min <= desc["area"] <= area_max,
    }
    fallas = [motivo for motivo, ok in reglas.items() if not ok]
    return not fallas, 1 - len(fallas) / len(reglas), fallas


def detectar_balon_clasico(img: np.ndarray, rangos_candidato=("Balón (amarillo)", "Balón (azul)"),
                           rangos_color=("Balón (amarillo)",), tam_cierre: int = 7,
                           area_min: float = 15, **reglas) -> list[dict]:
    """Busca el balón con color + morfología + forma.

    1. Máscara de candidatos: unión de los rangos HSV (por defecto amarillo y azul,
       así también aparecen las camisetas azules como candidatos a descartar).
    2. Cierre + relleno de huecos para unir las franjas del balón.
    3. Descriptores de cada componente y clasificación con :func:`es_balon`.

    Returns:
        Lista de candidatos (descriptores + ``es_balon``, ``puntaje``, ``motivos``,
        ``frac_color``), primero los clasificados como balón y luego por puntaje.
    """
    mascara = np.zeros(img.shape[:2], np.uint8)
    for r in rangos_candidato:
        mascara |= um.mascara_por_nombre(img, r)
    mascara = mo.limpiar_mascara(mascara, 0, tam_cierre, "Elipse", area_min=int(area_min), rellenar=True)
    candidatos = []
    for desc in describir_mascara(mascara, area_min):
        fc = fraccion_color(img, desc["contorno"], list(rangos_color))
        ok, puntaje, motivos = es_balon(desc, fc, area_min=area_min, **reglas)
        candidatos.append({**desc, "es_balon": ok, "puntaje": puntaje, "motivos": motivos, "frac_color": fc})
    candidatos.sort(key=lambda c: (c["es_balon"], c["puntaje"], c["circularidad"]), reverse=True)
    return candidatos


def mascara_candidatos_balon(img: np.ndarray, rangos=("Balón (amarillo)", "Balón (azul)"),
                             tam_cierre: int = 7) -> np.ndarray:
    """Máscara limpia de candidatos a balón (útil para mostrar el paso intermedio)."""
    m = np.zeros(img.shape[:2], np.uint8)
    for r in rangos:
        m |= um.mascara_por_nombre(img, r)
    return mo.limpiar_mascara(m, 0, tam_cierre, "Elipse", rellenar=True)


# --------------------------------------------------------------------------
# Proyección a la cancha
# --------------------------------------------------------------------------
def punto_apoyo(caja) -> tuple[float, float]:
    """Centro del borde inferior de la caja: donde los pies tocan el piso."""
    x1, _, x2, y2 = caja
    return (x1 + x2) / 2.0, float(y2)


def proyectar_jugadores(cajas, hom: cal.HomografiaCancha, tolerancia_m: float = 1.5) -> dict:
    """Proyecta los puntos de apoyo de los jugadores a la cancha (metros).

    Returns:
        ``metros`` (N×2), ``dentro`` (bool por jugador) y ``puntos_imagen``.
    """
    pies = np.array([punto_apoyo(c) for c in cajas], float).reshape(-1, 2)
    metros = cal.imagen_a_cancha(pies, hom)
    dentro = cal.esta_en_cancha(metros, hom, tolerancia_m) if len(metros) else np.zeros(0, bool)
    return {"metros": metros, "dentro": dentro, "puntos_imagen": pies}


RED_M = 9.0          # la red, en metros desde la línea de fondo rival
LINEA_ATAQUE_M = 12.0  # línea de ataque del lado propio (3 m detrás de la red)


def zona_propia(y_m: float) -> str:
    """Zona del lado propio según la distancia a la red."""
    return "ataque (cerca de la red)" if y_m < LINEA_ATAQUE_M else "defensa (atrás)"


def seleccionar_equipo_propio(cajas, equipos=None, hom: cal.HomografiaCancha | None = None,
                              tol_red_m: float = 0.2, tol_lateral_m: float = 0.2, fondo_m: float = 5.5) -> dict:
    """Decide qué jugadores son del equipo propio (el que está de espaldas a la cámara).

    Args:
        cajas: cajas ``(x1, y1, x2, y2)`` de las personas encontradas.
        equipos: grupo de camiseta de cada caja (0/1, de :func:`histogramas.asignar_equipos`); solo se usa
            si no hay homografía.
        hom: homografía de la media cancha propia, si se marcaron las esquinas.
        tol_red_m: margen hacia el lado rival (las rivales que bloquean pisan a ~0,5 m de la red).
        tol_lateral_m: margen fuera de las líneas laterales (el primer árbitro, en su silla junto a la red,
            queda fuera). Detrás de la línea de fondo no hay margen: ahí, justo afuera de las esquinas, se
            paran los jueces de línea.
        fondo_m: cuánto detrás de la línea de fondo propia se acepta (ahí está quien saca).

    Returns:
        ``propio`` (bool por caja) y ``criterio`` (texto); con homografía también ``metros`` y ``dentro``.
    """
    cajas = list(cajas)
    if not cajas:
        return {"propio": np.zeros(0, bool), "criterio": "sin jugadores"}
    if hom is not None:
        proy = proyectar_jugadores(cajas, hom)
        x, y = proy["metros"][:, 0], proy["metros"][:, 1]
        tol_x = np.where(y <= hom.largo_m, tol_lateral_m, 0.0)
        propio = ((x >= -tol_x) & (x <= hom.ancho_m + tol_x)
                  & (y >= RED_M - tol_red_m) & (y <= hom.largo_m + fondo_m))
        return {"propio": propio, "criterio": "posición en la cancha (homografía)",
                "metros": proy["metros"], "dentro": proy["dentro"]}
    equipos = np.asarray(equipos if equipos is not None else [0] * len(cajas))
    if len(set(equipos.tolist())) < 2:
        return {"propio": np.ones(len(cajas), bool), "criterio": "un solo grupo de camisetas"}
    pies_y = np.array([punto_apoyo(c)[1] for c in cajas])
    cercano = max((0, 1), key=lambda e: np.median(pies_y[equipos == e]))
    return {"propio": equipos == cercano, "criterio": "color de camiseta + cercanía a la cámara"}
