"""Genera el tablero de calibración de data/samples/.

Uso:
    python scripts/generar_sinteticos.py

Crea:
* data/samples/tablero/vista_XX.jpg: fotos sintéticas del tablero 9×6 con distorsión conocida
  (para la pestaña de calibración; con tu cámara, fotografía el tablero impreso).
* data/samples/tablero_plano.png: el patrón para imprimir.

Las escenas de voleibol de la app son fotos reales (``data/samples/reales/``). Las escenas
sintéticas con posiciones conocidas solo se generan en memoria para las pruebas automáticas.
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
    print(f"Muestras generadas en {DIR_MUESTRAS}")


if __name__ == "__main__":
    main()
