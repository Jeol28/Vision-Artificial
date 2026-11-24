"""Punto de entrada de VolleyVision (local y Hugging Face Spaces con ZeroGPU).

    python app.py            # abre http://127.0.0.1:7860

En un Space con ZeroGPU, ``spaces`` debe importarse antes que torch y los
modelos se cargan aquí, al arrancar (ver ``volleyvision/ui/modelos.py``).
"""

try:
    import spaces  # noqa: F401  (debe ir antes que torch en ZeroGPU)
except ImportError:
    pass

from volleyvision.ui import modelos  # noqa: E402
from volleyvision.ui.app import construir_app, lanzar  # noqa: E402

if modelos.EN_ZEROGPU:
    modelos.precargar()

demo = construir_app()

if __name__ == "__main__":
    lanzar(demo)
