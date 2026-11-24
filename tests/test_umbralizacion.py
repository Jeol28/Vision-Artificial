import cv2
import numpy as np
import pytest

from volleyvision.core import umbralizacion as um


@pytest.mark.parametrize("semilla", range(5))
def test_otsu_manual_igual_a_opencv(semilla):
    rng = np.random.default_rng(semilla)
    a = rng.normal(rng.uniform(40, 100), 12, 4000)
    b = rng.normal(rng.uniform(150, 210), 15, 3000)
    img = np.clip(np.concatenate([a, b]), 0, 255).astype(np.uint8).reshape(70, 100)
    T_manual, curva = um.otsu_manual(img)
    T_ocv, _ = um.otsu_opencv(img)
    assert abs(T_manual - T_ocv) <= 1
    assert curva.shape == (256,)
    assert curva[T_manual] == pytest.approx(curva.max())


def test_otsu_en_escena(escena):
    T_manual, _ = um.otsu_manual(escena.imagen)
    T_ocv, _ = um.otsu_opencv(escena.imagen)
    assert abs(T_manual - T_ocv) <= 1


def test_umbral_global_igual_a_threshold(imagen_gris):
    _, ocv = cv2.threshold(imagen_gris, 120, 255, cv2.THRESH_BINARY)
    np.testing.assert_array_equal(um.umbral_global(imagen_gris, 120), ocv)
    np.testing.assert_array_equal(um.umbral_global(imagen_gris, 120, invertir=True), 255 - ocv)


def test_hsv_aisla_el_balon(escena):
    m = um.mascara_por_nombre(escena.imagen, "Balón (amarillo)")
    cx, cy = escena.balon["centro"]
    r = escena.balon["radio"]
    ventana = m[cy - r:cy + r + 1, cx - r:cx + r + 1]
    assert (ventana > 0).mean() > 0.4
    # casi nada fuera del balón
    fuera = m.copy()
    fuera[cy - r - 3:cy + r + 4, cx - r - 3:cx + r + 4] = 0
    assert (fuera > 0).sum() < 0.002 * m.size


def test_hsv_rango_circular():
    img = np.zeros((1, 3, 3), np.uint8)
    img[0, 0] = (0, 0, 255)      # rojo puro H=0
    img[0, 1] = (40, 0, 255)     # rojo-magenta H≈175
    img[0, 2] = (0, 255, 0)      # verde
    m = um.umbral_hsv(img, (170, 100, 100), (5, 255, 255))
    assert list(m[0] > 0) == [True, True, False]


def test_adaptativo_tolera_iluminacion():
    # texto oscuro sobre fondo con gradiente fuerte: el global falla, el adaptativo no
    fondo = np.tile(np.linspace(40, 230, 300), (100, 1))
    img = fondo.copy()
    img[40:60, ::10] -= 35
    img = np.clip(img, 0, 255).astype(np.uint8)
    marcas = np.zeros(img.shape, bool)
    marcas[40:60, ::10] = True
    ad = um.umbral_adaptativo(img, "media", 15, 10, invertir=True) > 0
    _, glob = um.otsu_opencv(img, invertir=True)
    f1 = lambda pred: 2 * (pred & marcas).sum() / (pred.sum() + marcas.sum())  # noqa: E731
    assert f1(ad) > f1(glob > 0)
