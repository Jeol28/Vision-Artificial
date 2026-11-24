import numpy as np
import pytest

from volleyvision.core import filtros


@pytest.mark.parametrize("nombre", list(filtros.KERNELS_PREDEFINIDOS))
def test_convolucion_manual_igual_a_filter2d(imagen_gris, nombre):
    k = np.array(filtros.KERNELS_PREDEFINIDOS[nombre])
    res = filtros.comparar_convoluciones(imagen_gris, k)
    assert res["manual"].shape == imagen_gris.shape
    assert res["diferencia_max"] < 1e-9


def test_convolucion_manual_kernel_asimetrico_y_color(rng):
    img = rng.integers(0, 256, (40, 50, 3)).astype(np.uint8)
    k = rng.normal(size=(5, 3))
    manual = filtros.convolucion_manual(img, k)
    ocv = filtros.convolucion_opencv(img, k)
    np.testing.assert_allclose(manual, ocv, atol=1e-9)


def test_correlacion_vs_convolucion_difieren_en_kernel_asimetrico(imagen_gris):
    k = np.array(filtros.KERNELS_PREDEFINIDOS["Sobel X"])
    conv = filtros.convolucion_manual(imagen_gris, k, voltear=True)
    corr = filtros.convolucion_manual(imagen_gris, k, voltear=False)
    np.testing.assert_allclose(conv, -corr)


def test_kernel_desde_texto():
    k = filtros.kernel_desde_texto("1/9 1/9 1/9; 1/9, 1/9, 1/9\n1/9 1/9 1/9")
    assert k.shape == (3, 3)
    assert k.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        filtros.kernel_desde_texto("1 2\n3 4")


def test_mediana_elimina_sal_y_pimienta(imagen_gris):
    limpia = filtros.filtro_gaussiano(imagen_gris, 5)
    ruidosa = filtros.agregar_ruido(limpia, "sal_pimienta", 0.05)
    err_mediana = np.abs(filtros.filtro_mediana(ruidosa, 3).astype(int) - limpia).mean()
    err_promedio = np.abs(filtros.filtro_promedio(ruidosa, 3).astype(int) - limpia).mean()
    assert err_mediana < err_promedio


def test_resaltar_lineas_encuentra_segmentos(escena):
    res = filtros.resaltar_lineas_cancha(escena.imagen)
    assert len(res["segmentos"]) >= 4
    assert res["bordes"].dtype == np.uint8
