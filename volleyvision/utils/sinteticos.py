"""Generación de datos sintéticos con verdad conocida (*ground truth*).

* Vistas de un tablero de ajedrez tomadas por una cámara pinhole **con
  distorsión radial conocida**: permiten comprobar que la calibración recupera
  la matriz K y los coeficientes reales.
* Una escena de voleibol tomada **desde detrás de la línea de fondo propia**
  (cancha 9 × 18 m, red, 12 jugadores, el equipo propio de espaldas con su
  número, y un balón) con las posiciones reales de cada elemento: sirve para
  probar segmentación, caracterización y homografía sin depender de fotos.
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
# Escena de voleibol (cámara trasera)
# --------------------------------------------------------------------------
# Coordenadas de la cancha vista desde atrás: x de 0 a 9 m (izquierda → derecha),
# y de 0 m (línea de fondo rival) a 18 m (línea de fondo propia); red en y = 9.
# Mismo convenio que volleyvision.core.calibracion (utils no importa del core).
ESQUINAS_MEDIA_CANCHA_M = np.array([[0.0, 9.0], [9.0, 9.0], [9.0, 18.0], [0.0, 18.0]], np.float32)


@dataclass
class EscenaSintetica:
    """Escena generada y su verdad conocida."""

    imagen: np.ndarray
    esquinas: np.ndarray            # 4×2: esquinas de la media cancha propia (red izq., red der., fondo der., fondo izq.)
    H_metros_a_imagen: np.ndarray   # homografía real cancha(m) → imagen
    jugadores: list[dict] = field(default_factory=list)  # pos_m, equipo, numero, caja
    balon: dict | None = None       # centro, radio, pos_m, altura_m


# Píxeles de las 4 esquinas de la media cancha propia vistas por la cámara trasera
# (elevada, como en una tribuna o un trípode alto: separa en la imagen las filas de jugadores)
ESQUINAS_ESCENA = np.array([[300, 250], [660, 250], [920, 590], [40, 590]], dtype=np.float32)
TAM_ESCENA = (960, 600)


def _homografia_escena(esquinas=ESQUINAS_ESCENA) -> np.ndarray:
    return cv2.getPerspectiveTransform(ESQUINAS_MEDIA_CANCHA_M, np.asarray(esquinas, np.float32))


def _proyectar(H, x, y):
    p = H @ np.array([x, y, 1.0])
    return p[:2] / p[2]


def _escala_local(H, x, y) -> float:
    """Píxeles por metro alrededor del punto (x, y) de la cancha (medidos a lo ancho)."""
    return float(np.linalg.norm(_proyectar(H, x + 0.5, y) - _proyectar(H, x - 0.5, y)))


def _fondo_escena(H, tam=TAM_ESCENA, semilla=0) -> np.ndarray:
    """Piso (cancha + zona libre) en perspectiva y pared del coliseo al fondo."""
    rng = np.random.default_rng(semilla)
    W, Hh = tam
    ppm = 40
    m0 = 4.0  # metros de zona libre alrededor de la cancha
    lienzo = np.zeros((int((LARGO + 2 * m0) * ppm), int((ANCHO + 2 * m0) * ppm), 3), np.uint8)
    lienzo[:] = COLOR_ZONA_LIBRE

    def p(x, y):
        return int((x + m0) * ppm), int((y + m0) * ppm)

    cv2.rectangle(lienzo, p(0, 0), p(ANCHO, LARGO), COLOR_CANCHA, -1)
    blanco = (240, 240, 240)
    cv2.rectangle(lienzo, p(0, 0), p(ANCHO, LARGO), blanco, 4)
    for y in (LARGO / 2 - 3, LARGO / 2, LARGO / 2 + 3):  # ataque rival, central, ataque propia
        cv2.line(lienzo, p(0, y), p(ANCHO, y), blanco, 4)
    S = np.array([[ppm, 0, m0 * ppm], [0, ppm, m0 * ppm], [0, 0, 1]], dtype=np.float64)
    # Pared/gradas del coliseo detrás de la cancha rival
    pared = np.zeros((Hh, W, 3), np.uint8)
    grad = np.linspace(40, 75, Hh)[:, None]
    pared[:] = np.stack([grad * 1.1, grad * 0.95, grad * 0.85], axis=-1).astype(np.uint8)
    borde_piso = int(_proyectar(H, ANCHO / 2, -m0)[1])
    for _ in range(6):
        x0 = int(rng.integers(0, W - 120))
        y0 = max(5, borde_piso - 50)
        cv2.rectangle(pared, (x0, y0), (x0 + int(rng.integers(60, 140)), y0 + 30),
                      tuple(int(v) for v in rng.integers(70, 140, 3)), -1)
    piso = cv2.warpPerspective(lienzo, H @ np.linalg.inv(S), tam, flags=cv2.INTER_LINEAR)
    mascara = cv2.warpPerspective(np.full(lienzo.shape[:2], 255, np.uint8), H @ np.linalg.inv(S), tam)
    return np.where(mascara[..., None] > 0, piso, pared)


def _dibujar_red(img, H) -> np.ndarray:
    """Red en el plano vertical y = 9 m (de 1.43 a 2.43 m de altura): desde atrás se ve de frente."""
    arriba, abajo = [], []
    for x in (-0.6, ANCHO + 0.6):
        base = _proyectar(H, x, LARGO / 2)
        s = _escala_local(H, x, LARGO / 2)
        arriba.append(base - [0, 2.43 * s])
        abajo.append(base - [0, 1.43 * s])
        cv2.line(img, tuple(base.astype(int)), tuple((base - [0, 2.55 * s]).astype(int)), (70, 70, 70), 4)
    capa = img.copy()
    for t in np.linspace(0, 1, 60):  # hilos verticales
        a = arriba[0] * (1 - t) + arriba[1] * t
        b = abajo[0] * (1 - t) + abajo[1] * t
        cv2.line(capa, tuple(a.astype(int)), tuple(b.astype(int)), (30, 30, 30), 1)
    for t in np.linspace(0, 1, 9):  # hilos horizontales
        a = arriba[0] * (1 - t) + abajo[0] * t
        b = arriba[1] * (1 - t) + abajo[1] * t
        cv2.line(capa, tuple(a.astype(int)), tuple(b.astype(int)), (30, 30, 30), 1)
    img = cv2.addWeighted(capa, 0.45, img, 0.55, 0)
    cv2.line(img, tuple(arriba[0].astype(int)), tuple(arriba[1].astype(int)), (235, 235, 235), 3)
    return img


# Formaciones escalonadas (tipo recepción en "W"): nadie queda justo detrás de otro
FORMACION_PROPIA = [(1.5, 10.9), (4.5, 11.4), (7.5, 10.9), (3.0, 14.6), (6.0, 14.6), (4.5, 16.6)]
FORMACION_RIVAL = [(7.8, 6.5), (4.6, 7.6), (1.4, 6.8), (6.2, 3.4), (3.0, 3.6), (4.6, 1.6)]


def _posiciones_jugadores(rng) -> list[dict]:
    """Equipo A (propio, de espaldas a la cámara) en y ∈ (9, 18); equipo B (rival) en y ∈ (0, 9)."""
    numeros = rng.choice(np.arange(1, 20), size=12, replace=False)
    jugadores = []
    for equipo, formacion in (("A", FORMACION_PROPIA), ("B", FORMACION_RIVAL)):
        for x, y in formacion:
            pos = (x + rng.uniform(-0.5, 0.5), y + rng.uniform(-0.5, 0.5))
            jugadores.append({"pos_m": pos, "equipo": equipo, "numero": int(numeros[len(jugadores)])})
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
    if jug["equipo"] == "A":  # equipo propio de espaldas: número en la camiseta
        texto = str(jug["numero"])
        escala = 0.28 * s / 22
        grosor = max(1, int(round(0.05 * s)))
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
        cv2.putText(img, texto, (int(cx - tw / 2), int(by - 1.20 * s)), cv2.FONT_HERSHEY_SIMPLEX, escala,
                    (240, 240, 240), grosor, cv2.LINE_AA)
    cv2.circle(img, (int(cx), int(by - 1.76 * s)), max(2, int(0.13 * s)), piel, -1)
    pelo = (int(cx), int(by - (1.78 if jug["equipo"] == "A" else 1.84) * s))  # de espaldas se ve más pelo
    cv2.circle(img, pelo, max(2, int((0.12 if jug["equipo"] == "A" else 0.10) * s)), (30, 30, 40), -1)
    return int(cx - 0.36 * s), int(by - 1.89 * s), int(cx + 0.36 * s), int(by)


def _dibujar_balon(img, H, pos_m, altura_m) -> dict:
    base = _proyectar(H, *pos_m)
    s = _escala_local(H, *pos_m)
    centro = base - [0, altura_m * s]
    radio = max(4, int(round(0.21 * s * 1.3)))  # radio ~2.6 veces el real (21 cm de diámetro) para que se vea
    c = tuple(int(v) for v in centro)
    cv2.circle(img, c, radio, COLOR_BALON, -1, cv2.LINE_AA)
    for ang in (20, 140, 260):
        cv2.ellipse(img, c, (radio, max(1, radio // 3)), ang, 0, 180, COLOR_FRANJA, max(1, radio // 7), cv2.LINE_AA)
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


def renderizar_escena(jugadores: list[dict], balon_pos=(7.6, 16.3), balon_altura=1.2,
                      semilla: int = 0, fondo: np.ndarray | None = None,
                      esquinas=ESQUINAS_ESCENA) -> EscenaSintetica:
    """Dibuja la escena vista desde atrás: rivales, red, equipo propio y balón (de atrás hacia adelante)."""
    rng = np.random.default_rng(semilla)
    H = _homografia_escena(esquinas)
    img = (fondo if fondo is not None else _fondo_escena(H, semilla=semilla)).copy()
    jugadores = [dict(j) for j in jugadores]
    orden = sorted(jugadores, key=lambda j: j["pos_m"][1])  # los más lejanos primero
    red_dibujada = False
    for j in orden:
        if not red_dibujada and j["pos_m"][1] > LARGO / 2:
            img = _dibujar_red(img, H)
            red_dibujada = True
        j["caja"] = _dibujar_jugador(img, H, j, rng)
    if not red_dibujada:
        img = _dibujar_red(img, H)
    balon = _dibujar_balon(img, H, balon_pos, balon_altura) if balon_pos is not None else None
    img = _iluminacion(img, rng)
    return EscenaSintetica(img, np.asarray(esquinas, np.float32), H, jugadores, balon)


def generar_cancha_sintetica(semilla: int = 0) -> EscenaSintetica:
    """Escena completa de voleibol tomada desde detrás de la línea de fondo propia."""
    rng = np.random.default_rng(semilla)
    return renderizar_escena(_posiciones_jugadores(rng), semilla=semilla)


def generar_video_sintetico(ruta: str | Path, n_frames: int = 60, fps: int = 20, semilla: int = 0) -> Path:
    """Video corto: el balón sale del campo propio, cruza la red y cae en el rival."""
    rng = np.random.default_rng(semilla)
    jugadores = _posiciones_jugadores(rng)
    H = _homografia_escena()
    fondo = _fondo_escena(H, semilla=semilla)
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    escritor = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), fps, TAM_ESCENA)
    for i in range(n_frames):
        t = i / (n_frames - 1)
        pos = (4.5 - 0.5 * t, 15.0 - 11.0 * t)
        altura = 1.2 + 6.0 * t * (1 - t) * 1.6
        movidos = [{**j, "pos_m": (j["pos_m"][0] + 0.3 * np.sin(i / 6 + k), j["pos_m"][1])}
                   for k, j in enumerate(jugadores)]
        escena = renderizar_escena(movidos, pos, altura, semilla=semilla + i, fondo=fondo)
        escritor.write(escena.imagen)
    escritor.release()
    return ruta
