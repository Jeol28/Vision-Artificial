import numpy as np
import pytest

from volleyvision.core import morfologia as mo
from volleyvision.core import umbralizacion as um


@pytest.fixture
def mascara_ruidosa(rng):
    m = np.zeros((120, 160), np.uint8)
    m[30:90, 40:110] = 255                 # objeto grande
    ruido = rng.random(m.shape) < 0.01     # sal fuera del objeto
    m[ruido] = 255
    m[55:58, 70:73] = 0                    # hueco pequeño dentro
    return m


@pytest.mark.parametrize("forma", list(mo.FORMAS))
@pytest.mark.parametrize("tam", [3, 5, 7])
def test_erosion_y_dilatacion_manual(mascara_ruidosa, forma, tam):
    ee = mo.elemento_estructurante(forma, tam)
    np.testing.assert_array_equal(mo.erosion_manual(mascara_ruidosa, ee), mo.erosion(mascara_ruidosa, ee))
    np.testing.assert_array_equal(mo.dilatacion_manual(mascara_ruidosa, ee), mo.dilatacion(mascara_ruidosa, ee))


def test_apertura_elimina_ruido_y_cierre_rellena(mascara_ruidosa):
    ee = mo.elemento_estructurante("Elipse", 5)
    abierta = mo.apertura(mascara_ruidosa, ee)
    assert mo.contar_componentes(abierta) == 1
    cerrada = mo.cierre(abierta, ee)
    assert cerrada[56, 71] == 255


def test_dualidad_erosion_dilatacion(mascara_ruidosa):
    ee = mo.elemento_estructurante("Rectángulo", 5)
    interior = (slice(5, -5), slice(5, -5))  # se ignora el efecto del borde
    izq = 255 - mo.erosion(mascara_ruidosa, ee)
    der = mo.dilatacion(255 - mascara_ruidosa, ee[::-1, ::-1])
    np.testing.assert_array_equal(izq[interior], der[interior])


def test_idempotencia_apertura(mascara_ruidosa):
    ee = mo.elemento_estructurante("Elipse", 5)
    una = mo.apertura(mascara_ruidosa, ee)
    np.testing.assert_array_equal(mo.apertura(una, ee), una)


def test_limpiar_mascara_del_balon(escena):
    m = um.mascara_por_nombre(escena.imagen, "Balón (amarillo)")
    # cierre de 11 px: une los trozos amarillos que separan las franjas azules del balón
    limpia = mo.limpiar_mascara(m, 3, 11, "Elipse", area_min=20, rellenar=True)
    assert mo.contar_componentes(limpia) == 1
    cx, cy = escena.balon["centro"]
    assert limpia[cy, cx] == 255
