"""Fixtures compartidas: imágenes sintéticas con verdad conocida."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from volleyvision.utils import sinteticos  # noqa: E402


@pytest.fixture(scope="session")
def escena():
    """Escena de voleibol sintética (imagen + posiciones reales)."""
    return sinteticos.generar_cancha_sintetica(semilla=0)


@pytest.fixture(scope="session")
def vistas_tablero():
    return sinteticos.generar_vistas_tablero(n=10, semilla=11)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def imagen_gris(rng):
    """Imagen de grises con estructura (gradiente + bloques) y ruido."""
    img = np.tile(np.linspace(0, 200, 128), (96, 1))
    img[20:60, 30:80] = 240
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
