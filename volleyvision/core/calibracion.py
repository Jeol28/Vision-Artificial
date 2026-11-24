r"""Modelos de cámara y calibración.

Modelo pinhole
--------------
Una cámara pinhole proyecta un punto 3D del mundo :math:`P_w = (X, Y, Z)` en un
píxel :math:`p = (u, v)` mediante

.. math::

    s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
    = K \, [R \mid t] \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix},
    \qquad
    K = \begin{bmatrix} f_x & \gamma & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}

* :math:`[R \mid t]` (parámetros **extrínsecos**) lleva el punto del sistema del
  mundo al de la cámara (rotación + traslación).
* :math:`K` (parámetros **intrínsecos**) contiene las distancias focales en
  píxeles :math:`f_x, f_y`, el punto principal :math:`(c_x, c_y)` y el sesgo
  :math:`\gamma` (≈ 0 en sensores modernos).
* :math:`s` es un factor de escala (la profundidad Z del punto en la cámara).

Distorsión de lente (modelo de Brown–Conrady)
---------------------------------------------
Las lentes reales curvan las rectas. Con :math:`(x, y)` coordenadas normalizadas
y :math:`r^2 = x^2 + y^2`:

.. math::

    x_d = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2 p_1 x y + p_2 (r^2 + 2x^2)

    y_d = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1 (r^2 + 2y^2) + 2 p_2 x y

:math:`k_i` son los coeficientes radiales (barril si :math:`k_1<0`, cojín si
:math:`k_1>0`) y :math:`p_i` los tangenciales. OpenCV los devuelve en el orden
``(k1, k2, p1, p2, k3)``.

Calibración de Zhang
--------------------
Con varias fotos de un patrón plano de geometría conocida (tablero de ajedrez,
Z = 0), cada vista da una homografía :math:`H = K [r_1\ r_2\ t]`. Las
restricciones de ortonormalidad de :math:`r_1, r_2` permiten despejar K en
forma cerrada, y luego Levenberg–Marquardt refina todos los parámetros
minimizando el **error de reproyección**:

.. math::

    E = \sqrt{\frac{1}{N} \sum_i \lVert p_i - \hat p_i(K, d, R, t) \rVert^2}

Homografía de la cancha (cámara trasera)
----------------------------------------
La cámara está detrás de la línea de fondo del equipo propio y ve las 4
esquinas de **la media cancha propia** (9 × 9 m): las dos de la línea central,
bajo la red, y las dos de la línea de fondo. El piso es un plano, así que la
relación entre sus coordenadas en metros y los píxeles de la imagen es una
**homografía** 3×3 (8 grados de libertad): esas 4 esquinas bastan para
resolverla con DLT. La homografía vale para **todo** el plano del piso, de modo
que también ubica a los rivales al otro lado de la red (con menos precisión,
porque están más lejos).

Coordenadas de la cancha completa (9 × 18 m), vistas desde atrás:
``x`` de 0 a 9 m de izquierda a derecha; ``y`` de 0 m (línea de fondo rival)
a 18 m (línea de fondo propia); la red está en ``y = 9``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ANCHO_CANCHA_M = 9.0
LARGO_CANCHA_M = 18.0
# Esquinas de la media cancha propia en el orden en que se marcan en la imagen
ORDEN_ESQUINAS = ("superior izquierda (red)", "superior derecha (red)",
                  "inferior derecha (línea de fondo)", "inferior izquierda (línea de fondo)")
ESQUINAS_MEDIA_CANCHA_M = np.array([[0.0, 9.0], [9.0, 9.0], [9.0, 18.0], [0.0, 18.0]], np.float32)


# --------------------------------------------------------------------------
# Calibración intrínseca
# --------------------------------------------------------------------------
@dataclass
class ResultadoCalibracion:
    """Resultado de :func:`calibrar_camara`."""

    K: np.ndarray                    # matriz intrínseca 3×3
    dist: np.ndarray                 # (k1, k2, p1, p2, k3)
    error_rms: float                 # error de reproyección global (px)
    errores_por_imagen: list[float]  # error medio de cada vista (px)
    rvecs: list[np.ndarray]
    tvecs: list[np.ndarray]
    tam_imagen: tuple[int, int]      # (ancho, alto)
    indices_validos: list[int]       # imágenes en las que se encontró el patrón


def puntos_objeto_tablero(patron: tuple[int, int], tam_cuadro: float = 1.0) -> np.ndarray:
    """Coordenadas 3D (Z = 0) de las esquinas interiores del tablero.

    Args:
        patron: (columnas, filas) de esquinas **interiores**.
        tam_cuadro: lado de cada cuadro en las unidades deseadas (mm, cm, ...).
    """
    cols, filas = patron
    obj = np.zeros((cols * filas, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:filas].T.reshape(-1, 2) * tam_cuadro
    return obj


def encontrar_esquinas(img: np.ndarray, patron: tuple[int, int]) -> tuple[bool, np.ndarray | None]:
    """Detecta y refina a nivel subpíxel las esquinas del tablero.

    ``cv2.findChessboardCorners`` localiza las esquinas a nivel de píxel y
    ``cv2.cornerSubPix`` las refina buscando el punto donde el gradiente es
    ortogonal al vector hacia cada vecino.
    """
    gris = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    banderas = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, esquinas = cv2.findChessboardCorners(gris, patron, banderas)
    if not ok:
        return False, None
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    esquinas = cv2.cornerSubPix(gris, esquinas, (11, 11), (-1, -1), criterio)
    return True, esquinas


def dibujar_esquinas(img: np.ndarray, patron, esquinas, encontrado: bool = True) -> np.ndarray:
    """Dibuja las esquinas detectadas (colores por fila, como en OpenCV)."""
    salida = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if esquinas is not None:
        cv2.drawChessboardCorners(salida, patron, esquinas, encontrado)
    return salida


def calibrar_camara(imagenes: list[np.ndarray], patron: tuple[int, int] = (9, 6),
                    tam_cuadro: float = 25.0) -> ResultadoCalibracion:
    """Calibra la cámara (método de Zhang) a partir de fotos de un tablero.

    Returns:
        :class:`ResultadoCalibracion` con K, distorsión y errores de reproyección.

    Raises:
        ValueError: si el patrón se detecta en menos de 3 imágenes (se necesitan
            al menos 3 vistas para resolver los 5 intrínsecos de K).
    """
    obj = puntos_objeto_tablero(patron, tam_cuadro)
    pts_obj, pts_img, validos = [], [], []
    tam = None
    for i, img in enumerate(imagenes):
        ok, esquinas = encontrar_esquinas(img, patron)
        if ok:
            pts_obj.append(obj)
            pts_img.append(esquinas)
            validos.append(i)
            tam = (img.shape[1], img.shape[0])
    if len(validos) < 3:
        raise ValueError(f"El patrón solo se encontró en {len(validos)} imágenes; se necesitan al menos 3.")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(pts_obj, pts_img, tam, None, None)
    errores = [error_reproyeccion(o, p, r, t, K, dist) for o, p, r, t in zip(pts_obj, pts_img, rvecs, tvecs)]
    return ResultadoCalibracion(K, dist.ravel(), float(rms), errores, list(rvecs), list(tvecs), tam, validos)


def error_reproyeccion(obj: np.ndarray, img_pts: np.ndarray, rvec, tvec, K, dist) -> float:
    """Error RMS (px) entre las esquinas detectadas y las reproyectadas con el modelo."""
    proyectados, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    dif = img_pts.reshape(-1, 2) - proyectados.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(dif ** 2, axis=1))))


def proyectar_pinhole(puntos_3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Proyección pinhole implementada a mano (sin distorsión): p ~ K (R X + t).

    Se usa para explicar el modelo y en las pruebas, comparándola con ``cv2.projectPoints``.
    """
    cam = (R @ np.asarray(puntos_3d, float).T).T + np.asarray(t, float).reshape(1, 3)
    hom = (K @ cam.T).T
    return hom[:, :2] / hom[:, 2:3]


def corregir_distorsion(img: np.ndarray, K: np.ndarray, dist: np.ndarray,
                        alfa: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Elimina la distorsión de lente.

    Args:
        alfa: 0 recorta a solo píxeles válidos; 1 conserva toda la imagen original
            (con bordes negros). Ver ``cv2.getOptimalNewCameraMatrix``.

    Returns:
        (imagen corregida, nueva matriz K).
    """
    h, w = img.shape[:2]
    K_nueva, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alfa, (w, h))
    return cv2.undistort(img, K, dist, None, K_nueva), K_nueva


# --------------------------------------------------------------------------
# Homografía de la cancha
# --------------------------------------------------------------------------
@dataclass
class HomografiaCancha:
    """Homografías imagen ↔ cancha.

    * ``H_metros``: píxel de la imagen → coordenadas de la cancha en metros.
    * ``H_pixeles``: píxel de la imagen → píxel de la vista cenital.
    """

    H_metros: np.ndarray
    H_pixeles: np.ndarray
    tam_salida: tuple[int, int]
    px_por_m: float
    margen_m: float
    largo_m: float = LARGO_CANCHA_M
    ancho_m: float = ANCHO_CANCHA_M


def homografia_dlt(origen: np.ndarray, destino: np.ndarray) -> np.ndarray:
    r"""Estima la homografía con la *Direct Linear Transform* (implementación propia).

    Cada correspondencia :math:`(x, y) \to (x', y')` aporta dos ecuaciones lineales en
    los 9 elementos de H:

    .. math::

        \begin{bmatrix} -x & -y & -1 & 0 & 0 & 0 & x x' & y x' & x' \\
                        0 & 0 & 0 & -x & -y & -1 & x y' & y y' & y' \end{bmatrix} h = 0

    Se apilan las ecuaciones en :math:`A h = 0` y la solución (con :math:`\lVert h
    \rVert = 1`) es el vector singular derecho de A con menor valor singular.
    Los puntos se normalizan antes (Hartley) para mejorar el condicionamiento.
    """
    origen = np.asarray(origen, float)
    destino = np.asarray(destino, float)
    if len(origen) < 4:
        raise ValueError("Se necesitan al menos 4 correspondencias.")

    def normalizar(p):
        c = p.mean(axis=0)
        d = np.mean(np.linalg.norm(p - c, axis=1))
        s = np.sqrt(2) / d if d > 0 else 1.0
        T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
        ph = np.column_stack([p, np.ones(len(p))]) @ T.T
        return ph[:, :2], T

    o, To = normalizar(origen)
    d, Td = normalizar(destino)
    filas = []
    for (x, y), (xp, yp) in zip(o, d):
        filas.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        filas.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    _, _, Vt = np.linalg.svd(np.asarray(filas))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ To
    return H / H[2, 2]


def homografia_cancha(esquinas_img, px_por_m: float = 30.0, margen_m: float = 2.0) -> HomografiaCancha:
    """Calcula la homografía de la cancha a partir de las 4 esquinas de la media cancha propia.

    Args:
        esquinas_img: 4 puntos de la imagen tomada desde atrás, en el orden
            superior izquierda (red), superior derecha (red), inferior derecha
            (línea de fondo) e inferior izquierda (línea de fondo). Corresponden a
            (0, 9), (9, 9), (9, 18) y (0, 18) m (:data:`ESQUINAS_MEDIA_CANCHA_M`).
        px_por_m: resolución de la vista cenital.
        margen_m: zona libre (en metros) que se conserva alrededor de la cancha.

    La vista cenital cubre la cancha completa (9 × 18 m) con la red horizontal y el
    campo propio abajo, como la ve la cámara.
    """
    esquinas = np.asarray(esquinas_img, np.float32).reshape(4, 2)
    H_m = cv2.getPerspectiveTransform(esquinas, ESQUINAS_MEDIA_CANCHA_M)
    S = np.array([[px_por_m, 0, margen_m * px_por_m], [0, px_por_m, margen_m * px_por_m], [0, 0, 1]])
    tam = (int(round((ANCHO_CANCHA_M + 2 * margen_m) * px_por_m)),
           int(round((LARGO_CANCHA_M + 2 * margen_m) * px_por_m)))
    return HomografiaCancha(H_m, S @ H_m, tam, px_por_m, margen_m)


def vista_cenital(img: np.ndarray, hom: HomografiaCancha, dibujar_lineas: bool = True) -> np.ndarray:
    """Rectifica la imagen para ver la cancha desde arriba (``cv2.warpPerspective``)."""
    salida = cv2.warpPerspective(img, hom.H_pixeles, hom.tam_salida, flags=cv2.INTER_LINEAR)
    if dibujar_lineas:
        p = hom.px_por_m
        m = hom.margen_m

        def px(x, y):
            return int(round((x + m) * p)), int(round((y + m) * p))

        naranja = (53, 107, 255)
        cv2.rectangle(salida, px(0, 0), px(hom.ancho_m, hom.largo_m), naranja, 1)
        cv2.line(salida, px(0, hom.largo_m / 2), px(hom.ancho_m, hom.largo_m / 2), naranja, 1)  # red
    return salida


def proyectar_puntos(puntos, H: np.ndarray) -> np.ndarray:
    """Aplica una homografía a un conjunto de puntos (N×2) con ``cv2.perspectiveTransform``."""
    pts = np.asarray(puntos, np.float64).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def imagen_a_cancha(puntos, hom: HomografiaCancha) -> np.ndarray:
    """Convierte píxeles de la imagen a metros sobre la cancha."""
    return proyectar_puntos(puntos, hom.H_metros)


def esta_en_cancha(puntos_m, hom: HomografiaCancha, tolerancia_m: float = 1.0) -> np.ndarray:
    """Indica qué puntos (en metros) caen dentro de la cancha más una tolerancia."""
    p = np.asarray(puntos_m, float).reshape(-1, 2)
    t = tolerancia_m
    return (p[:, 0] >= -t) & (p[:, 0] <= hom.ancho_m + t) & (p[:, 1] >= -t) & (p[:, 1] <= hom.largo_m + t)
