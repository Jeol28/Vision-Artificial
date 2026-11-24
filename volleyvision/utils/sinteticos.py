"""Generación de datos sintéticos con verdad conocida (*ground truth*).

* Vistas de un tablero de ajedrez tomadas por una cámara pinhole **con
  distorsión radial conocida**: permiten comprobar que la calibración recupera
  la matriz K y los coeficientes reales.
* Una escena de voleibol en perspectiva (cancha 9 × 18 m, red, 12 jugadores de
  dos equipos y un balón) con las posiciones reales de cada elemento: sirve
  para probar segmentación, caracterización y homografía sin depender de fotos.
* Un video corto de la misma escena con el balón en vuelo.

Los colores son genéricos (cancha naranja, zona libre verde, equipos rojo y
azul, balón amarillo con franjas azules) y no reproducen ninguna marca.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Cámara "real" usada para sintetizar el tablero
K_VERDADERA = np.array([[620.0, 0.0, 322.0], [0.0, 615.0, 238.0], [0.0, 0.0, 1.0]])
DIST_VERDADERA = np.array([-0.28, 0.09, 0.0005, -0.0004, 0.0])
TAM_VISTA = (640, 480)
PATRON = (9, 6)  # esquinas interiores (columnas, filas)
TAM_CUADRO_MM = 25.0

LARGO, ANCHO = 18.0, 9.0
COLOR_CANCHA = (60, 130, 230)
COLOR_ZONA_LIBRE = (80, 140, 50)
COLOR_EQUIPO = {"A": (40, 40, 200), "B": (180, 80, 30)}
COLOR_BALON = (0, 215, 255)
COLOR_FRANJA = (160, 70, 20)


# --------------------------------------------------------------------------
# Tablero de ajedrez
# --------------------------------------------------------------------------
def generar_tablero(patron=PATRON, px_cuadro: int = 60, margen: int = 60) -> np.ndarray:
    """Tablero plano en escala de grises con ``patron`` esquinas interiores."""
    cols, filas = patron[0] + 1, patron[1] + 1
    img = np.full((filas * px_cuadro + 2 * margen, cols * px_cuadro + 2 * margen), 255, np.uint8)
    for f in range(filas):
        for c in range(cols):
            if (f + c) % 2 == 0:
                y, x = margen + f * px_cuadro, margen + c * px_cuadro
                img[y:y + px_cuadro, x:x + px_cuadro] = 0
    return img


def distorsionar(img: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Aplica la distorsión de lente del modelo de Brown–Conrady a una imagen ideal.

    Para cada píxel de la imagen distorsionada se calcula (con
    ``cv2.undistortPoints``) en qué punto de la imagen ideal cae, y se muestrea
    ahí con ``cv2.remap``.
    """
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
    ideal = cv2.undistortPoints(pts, K, dist, P=K).reshape(h, w, 2)
    return cv2.remap(img, ideal[..., 0], ideal[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=128)


def generar_vistas_tablero(n: int = 14, semilla: int = 3, K=K_VERDADERA, dist=DIST_VERDADERA,
                           tam=TAM_VISTA, patron=PATRON) -> list[np.ndarray]:
    """Genera ``n`` fotos del tablero desde poses aleatorias con distorsión conocida."""
    rng = np.random.default_rng(semilla)
    px, margen = 60, 60
    plano = generar_tablero(patron, px, margen)
    s = TAM_CUADRO_MM
    cx = (patron[0] + 1) * s / 2
    cy = (patron[1] + 1) * s / 2
    # píxel del tablero → coordenadas del plano (mm) centradas
    A = np.array([[s / px, 0, -margen * s / px - cx], [0, s / px, -margen * s / px - cy], [0, 0, 1]])
    vistas = []
    while len(vistas) < n:
        ang = np.deg2rad([rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(-15, 15)])
        R, _ = cv2.Rodrigues(ang.reshape(3, 1))
        t = np.array([rng.uniform(-45, 45), rng.uniform(-35, 35), rng.uniform(380, 520)])
        H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ A
        fondo = np.clip(rng.normal(110, 12, (tam[1], tam[0])), 0, 255).astype(np.uint8)
        ideal = cv2.warpPerspective(plano, H, tam, borderMode=cv2.BORDER_TRANSPARENT, dst=fondo)
        img = distorsionar(ideal, K, dist)
        img = cv2.GaussianBlur(img, (3, 3), 0.7)
        img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
        ok, _ = cv2.findChessboardCorners(img, patron)
        if ok:
            vistas.append(img)
    return vistas


# --------------------------------------------------------------------------
# Escena de voleibol
# --------------------------------------------------------------------------
@dataclass
class EscenaSintetica:
    """Escena generada y su verdad conocida."""

    imagen: np.ndarray
    esquinas: np.ndarray            # 4×2: sup-izq, sup-der, inf-der, inf-izq (píxeles)
    H_metros_a_imagen: np.ndarray   # homografía real cancha(m) → imagen
    jugadores: list[dict] = field(default_factory=list)  # pos_m, equipo, caja
    balon: dict | None = None       # centro, radio, pos_m, altura_m


ESQUINAS_ESCENA = np.array([[250, 260], [710, 260], [900, 540], [60, 540]], dtype=np.float32)
TAM_ESCENA = (960, 600)


def _homografia_escena(esquinas=ESQUINAS_ESCENA) -> np.ndarray:
    metros = np.array([[0, 0], [LARGO, 0], [LARGO, ANCHO], [0, ANCHO]], dtype=np.float32)
    return cv2.getPerspectiveTransform(metros, np.asarray(esquinas, np.float32))


def _proyectar(H, x, y):
    p = H @ np.array([x, y, 1.0])
    return p[:2] / p[2]


def _escala_local(H, x, y) -> float:
    """Píxeles por metro alrededor del punto (x, y) de la cancha."""
    return float(np.linalg.norm(_proyectar(H, x + 0.5, y) - _proyectar(H, x - 0.5, y)))


def _fondo_escena(H, tam=TAM_ESCENA, semilla=0) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    W, Hh = tam
    ppm = 40
    m0 = 4.0  # metros de zona libre alrededor de la cancha
    lienzo = np.zeros((int((ANCHO + 2 * m0) * ppm), int((LARGO + 2 * m0) * ppm), 3), np.uint8)
    lienzo[:] = COLOR_ZONA_LIBRE

    def p(x, y):
        return int((x + m0) * ppm), int((y + m0) * ppm)

    cv2.rectangle(lienzo, p(0, 0), p(LARGO, ANCHO), COLOR_CANCHA, -1)
    blanco = (240, 240, 240)
    cv2.rectangle(lienzo, p(0, 0), p(LARGO, ANCHO), blanco, 4)
    for x in (LARGO / 2 - 3, LARGO / 2, LARGO / 2 + 3):
        cv2.line(lienzo, p(x, 0), p(x, ANCHO), blanco, 4)
    S = np.array([[ppm, 0, m0 * ppm], [0, ppm, m0 * ppm], [0, 0, 1]], dtype=np.float64)
    # Pared/gradas del coliseo
    pared = np.zeros((Hh, W, 3), np.uint8)
    grad = np.linspace(40, 75, Hh)[:, None]
    pared[:] = np.stack([grad * 1.1, grad * 0.95, grad * 0.85], axis=-1).astype(np.uint8)
    for _ in range(7):
        x0 = int(rng.integers(0, W - 120))
        cv2.rectangle(pared, (x0, 150), (x0 + int(rng.integers(60, 140)), 185),
                      tuple(int(v) for v in rng.integers(70, 140, 3)), -1)
    piso = cv2.warpPerspective(lienzo, H @ np.linalg.inv(S), tam, flags=cv2.INTER_LINEAR)
    mascara = cv2.warpPerspective(np.full(lienzo.shape[:2], 255, np.uint8), H @ np.linalg.inv(S), tam)
    salida = np.where(mascara[..., None] > 0, piso, pared)
    # Red en x = 9 m (plano vertical): de 1.43 m a 2.43 m de altura
    arriba, abajo = [], []
    for y in (-0.6, ANCHO + 0.6):
        base = _proyectar(H, LARGO / 2, y)
        s = _escala_local(H, LARGO / 2, y)
        arriba.append(base - [0, 2.43 * s])
        abajo.append(base - [0, 1.43 * s])
        cv2.line(salida, tuple(base.astype(int)), tuple((base - [0, 2.55 * s]).astype(int)), (70, 70, 70), 4)
    capa = salida.copy()
    for t in np.linspace(0, 1, 40):
        a = arriba[0] * (1 - t) + arriba[1] * t
        b = abajo[0] * (1 - t) + abajo[1] * t
        cv2.line(capa, tuple(a.astype(int)), tuple(b.astype(int)), (30, 30, 30), 1)
    for t in np.linspace(0, 1, 8):
        a = arriba[0] * (1 - t) + abajo[0] * t
        b = arriba[1] * (1 - t) + abajo[1] * t
        cv2.line(capa, tuple(a.astype(int)), tuple(b.astype(int)), (30, 30, 30), 1)
    salida = cv2.addWeighted(capa, 0.6, salida, 0.4, 0)
    cv2.line(salida, tuple(arriba[0].astype(int)), tuple(arriba[1].astype(int)), (235, 235, 235), 3)
    return salida


def _posiciones_jugadores(rng) -> list[dict]:
    base = [(7.0, 1.6), (7.2, 4.5), (7.0, 7.4), (2.8, 1.8), (3.2, 4.6), (2.6, 7.2)]
    jugadores = []
    for equipo in ("A", "B"):
        for x, y in base:
            x += rng.uniform(-0.6, 0.6)
            y += rng.uniform(-0.6, 0.6)
            jugadores.append({"pos_m": (x if equipo == "A" else LARGO - x, y), "equipo": equipo})
    return jugadores


def _dibujar_jugador(img, H, jug, rng) -> tuple[int, int, int, int]:
    x, y = jug["pos_m"]
    pie = _proyectar(H, x, y)
    s = _escala_local(H, x, y)
    piel = (120, 160, 205)
    camiseta = COLOR_EQUIPO[jug["equipo"]]
    short = (35, 35, 35)
    cx, by = pie

    def r(xm0, ym0, xm1, ym1, color):
        cv2.rectangle(img, (int(cx + xm0 * s), int(by - ym1 * s)), (int(cx + xm1 * s), int(by - ym0 * s)), color, -1)

    r(-0.20, 0.0, -0.05, 0.85, piel)        # pierna izquierda
    r(0.05, 0.0, 0.20, 0.85, piel)          # pierna derecha
    r(-0.22, 0.0, -0.04, 0.10, (230, 230, 230))  # zapatillas
    r(0.04, 0.0, 0.22, 0.10, (230, 230, 230))
    r(-0.23, 0.82, 0.23, 1.05, short)       # pantaloneta
    r(-0.25, 1.03, 0.25, 1.62, camiseta)    # camiseta
    brazo = rng.uniform(0.95, 1.55)
    r(-0.36, brazo, -0.25, 1.60, piel)
    r(0.25, brazo, 0.36, 1.60, piel)
    cv2.circle(img, (int(cx), int(by - 1.76 * s)), max(2, int(0.13 * s)), piel, -1)
    cv2.circle(img, (int(cx), int(by - 1.82 * s)), max(2, int(0.10 * s)), (30, 30, 40), -1)
    return int(cx - 0.36 * s), int(by - 1.89 * s), int(cx + 0.36 * s), int(by)


def _dibujar_balon(img, H, pos_m, altura_m) -> dict:
    base = _proyectar(H, *pos_m)
    s = _escala_local(H, *pos_m)
    centro = base - [0, altura_m * s]
    radio = max(4, int(round(0.21 * s * 1.3)))  # diámetro real ~21 cm (ligeramente exagerado)
    c = tuple(int(v) for v in centro)
    cv2.circle(img, c, radio, COLOR_BALON, -1, cv2.LINE_AA)
    for ang in (20, 140, 260):
        cv2.ellipse(img, c, (radio, max(1, radio // 3)), ang, 0, 180, COLOR_FRANJA, max(1, radio // 5), cv2.LINE_AA)
    cv2.circle(img, (c[0] - radio // 3, c[1] - radio // 3), max(1, radio // 4), (180, 245, 255), -1, cv2.LINE_AA)
    return {"centro": c, "radio": radio, "pos_m": tuple(pos_m), "altura_m": altura_m}


def _iluminacion(img, rng) -> np.ndarray:
    """Simula la iluminación irregular de un coliseo (gradiente + viñeta + ruido)."""
    h, w = img.shape[:2]
    xs = np.linspace(-1, 1, w)[None, :]
    ys = np.linspace(-1, 1, h)[:, None]
    factor = (1.05 - 0.35 * (xs + 1) / 2) * (1 - 0.25 * (xs ** 2 + ys ** 2))
    out = img.astype(np.float32) * factor[..., None]
    out += rng.normal(0, 4, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def renderizar_escena(jugadores: list[dict], balon_pos=(12.0, 4.0), balon_altura=3.0,
                      semilla: int = 0, fondo: np.ndarray | None = None,
                      esquinas=ESQUINAS_ESCENA) -> EscenaSintetica:
    """Dibuja la escena con los jugadores y el balón indicados."""
    rng = np.random.default_rng(semilla)
    H = _homografia_escena(esquinas)
    img = (fondo if fondo is not None else _fondo_escena(H, semilla=semilla)).copy()
    jugadores = [dict(j) for j in jugadores]
    for j in sorted(jugadores, key=lambda j: j["pos_m"][1]):  # los del fondo primero
        j["caja"] = _dibujar_jugador(img, H, j, rng)
    balon = _dibujar_balon(img, H, balon_pos, balon_altura) if balon_pos is not None else None
    img = _iluminacion(img, rng)
    return EscenaSintetica(img, np.asarray(esquinas, np.float32), H, jugadores, balon)


def generar_cancha_sintetica(semilla: int = 0) -> EscenaSintetica:
    """Escena completa de voleibol en perspectiva lista para analizar."""
    rng = np.random.default_rng(semilla)
    return renderizar_escena(_posiciones_jugadores(rng), semilla=semilla)


def generar_video_sintetico(ruta: str | Path, n_frames: int = 60, fps: int = 20, semilla: int = 0) -> Path:
    """Video corto: el balón cruza la red en una parábola de la cancha A a la B."""
    rng = np.random.default_rng(semilla)
    jugadores = _posiciones_jugadores(rng)
    H = _homografia_escena()
    fondo = _fondo_escena(H, semilla=semilla)
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    escritor = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), fps, TAM_ESCENA)
    for i in range(n_frames):
        t = i / (n_frames - 1)
        pos = (4.0 + 10.0 * t, 3.0 + 3.0 * t)
        altura = 1.5 + 6.0 * t * (1 - t) * 1.6
        movidos = [{**j, "pos_m": (j["pos_m"][0] + 0.3 * np.sin(i / 6 + k), j["pos_m"][1])}
                   for k, j in enumerate(jugadores)]
        escena = renderizar_escena(movidos, pos, altura, semilla=semilla + i, fondo=fondo)
        escritor.write(escena.imagen)
    escritor.release()
    return ruta
