"""Entrada/salida de imágenes y video.

Convención del proyecto: todas las imágenes viajan entre módulos como arreglos
NumPy ``uint8`` en orden de canales **BGR** (el de OpenCV). La conversión a RGB
se hace solo al momento de mostrar.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
DIR_MUESTRAS = RAIZ_PROYECTO / "data" / "samples"
DIR_TABLERO = DIR_MUESTRAS / "tablero"
EXTENSIONES_IMAGEN = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
EXTENSIONES_VIDEO = {".mp4", ".avi", ".mov", ".mkv"}


def cargar_imagen(fuente: str | Path | bytes, max_lado: int | None = None) -> np.ndarray:
    """Carga una imagen BGR desde una ruta o desde los bytes de un archivo subido.

    Args:
        fuente: ruta en disco o contenido binario (p. ej. ``UploadedFile.getvalue()``).
        max_lado: si se indica, reduce la imagen para que su lado mayor no lo supere
            (acelera el procesamiento en CPU sin deformar la imagen).
    """
    if isinstance(fuente, (bytes, bytearray)):
        datos = np.frombuffer(fuente, dtype=np.uint8)
        img = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(fuente), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen.")
    return redimensionar_max(img, max_lado) if max_lado else img


def redimensionar_max(img: np.ndarray, max_lado: int) -> np.ndarray:
    """Escala la imagen (solo reduce) para que su lado mayor sea ``max_lado``."""
    h, w = img.shape[:2]
    escala = max_lado / max(h, w)
    if escala >= 1:
        return img
    return cv2.resize(img, (round(w * escala), round(h * escala)), interpolation=cv2.INTER_AREA)


def guardar_imagen(ruta: str | Path, img: np.ndarray) -> None:
    """Guarda una imagen BGR creando la carpeta si no existe."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta), img)


def bgr_a_rgb(img: np.ndarray) -> np.ndarray:
    """Convierte BGR→RGB para mostrar (las imágenes de un canal se devuelven igual)."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def a_gris(img: np.ndarray) -> np.ndarray:
    """Convierte a escala de grises si la imagen es de color."""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def listar_muestras() -> dict[str, Path]:
    """Imágenes de ejemplo disponibles (sintéticas y descargadas), nombre → ruta."""
    rutas = sorted(DIR_MUESTRAS.glob("*")) + sorted((DIR_MUESTRAS / "descargadas").glob("*"))
    return {r.name: r for r in rutas if r.suffix.lower() in EXTENSIONES_IMAGEN}


def listar_videos() -> dict[str, Path]:
    """Videos de ejemplo disponibles, nombre → ruta."""
    rutas = sorted(DIR_MUESTRAS.glob("*")) + sorted((DIR_MUESTRAS / "descargadas").glob("*"))
    return {r.name: r for r in rutas if r.suffix.lower() in EXTENSIONES_VIDEO}


def listar_tableros() -> list[Path]:
    """Imágenes del tablero de ajedrez para calibración."""
    return sorted(p for p in DIR_TABLERO.glob("*") if p.suffix.lower() in EXTENSIONES_IMAGEN)


def info_video(ruta: str | Path) -> tuple[int, float]:
    """Devuelve (número de frames, fps) de un video."""
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video {ruta}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return n, fps


def leer_frame(ruta: str | Path, indice: int) -> np.ndarray:
    """Lee el frame ``indice`` de un video como imagen BGR."""
    cap = cv2.VideoCapture(str(ruta))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, indice))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"No se pudo leer el frame {indice} de {ruta}")
    return frame
