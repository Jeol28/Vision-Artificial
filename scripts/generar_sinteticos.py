"""Genera las muestras sintéticas de data/samples/.

Uso:
    python scripts/generar_sinteticos.py

Crea:
* data/samples/tablero/vista_XX.jpg: fotos sintéticas del tablero 9×6 con distorsión conocida.
* data/samples/tablero_plano.png: el patrón para imprimir.
* data/samples/cancha_sintetica.jpg y cancha_sintetica_2.jpg: escenas de voleibol.
* data/samples/rally_sintetico.mp4: video corto con el balón en vuelo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from volleyvision.utils import sinteticos as sint  # noqa: E402
from volleyvision.utils.io import DIR_MUESTRAS, DIR_TABLERO, guardar_imagen  # noqa: E402


def main() -> None:
    guardar_imagen(DIR_MUESTRAS / "tablero_plano.png", sint.generar_tablero())
    for i, vista in enumerate(sint.generar_vistas_tablero()):
        guardar_imagen(DIR_TABLERO / f"vista_{i:02d}.jpg", vista)
    for i, semilla in enumerate((0, 5)):
        nombre = "cancha_sintetica.jpg" if i == 0 else f"cancha_sintetica_{i + 1}.jpg"
        guardar_imagen(DIR_MUESTRAS / nombre, sint.generar_cancha_sintetica(semilla).imagen)
    sint.generar_video_sintetico(DIR_MUESTRAS / "rally_sintetico.mp4")
    print(f"Muestras generadas en {DIR_MUESTRAS}")


if __name__ == "__main__":
    main()
