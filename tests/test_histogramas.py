import cv2
import numpy as np

from volleyvision.core import histogramas as hi


def test_histograma_manual_igual_a_calchist(imagen_gris):
    ocv = hi.calcular_histograma(imagen_gris, "Gris")["Gris"]
    np.testing.assert_array_equal(hi.histograma_manual(imagen_gris), ocv)


def test_histograma_normalizado_suma_uno(escena):
    for canal, h in hi.calcular_histograma(escena.imagen, "HSV", normalizar=True).items():
        assert abs(h.sum() - 1) < 1e-6, canal


def test_ecualizacion_manual_igual_a_opencv(imagen_gris):
    np.testing.assert_array_equal(hi.ecualizar_manual(imagen_gris), cv2.equalizeHist(imagen_gris))


def test_ecualizacion_aumenta_contraste():
    oscura = (np.tile(np.linspace(60, 120, 200), (100, 1))).astype(np.uint8)
    assert hi.contraste_rms(hi.ecualizar_global(oscura)) > hi.contraste_rms(oscura)


def test_clahe_mejora_uniformidad(escena):
    antes = hi.uniformidad_iluminacion(escena.imagen)
    despues = hi.uniformidad_iluminacion(hi.aplicar_clahe(escena.imagen, 3.0, 8))
    assert despues < antes


def test_asignar_equipos_separa_uniformes(escena):
    cajas = [j["caja"] for j in escena.jugadores]
    reales = [0 if j["equipo"] == "A" else 1 for j in escena.jugadores]
    hists = hi.histogramas_de_cajas(escena.imagen, cajas)
    equipos, M = hi.asignar_equipos(hists)
    assert M.shape == (12, 12)
    acierto = max(np.mean(np.array(equipos) == reales), np.mean(np.array(equipos) != reales))
    assert acierto == 1.0


def test_distancia_propia_es_cero(escena):
    h = hi.histograma_hs(escena.imagen)
    assert hi.comparar_histogramas(h, h, "Bhattacharyya") < 1e-6
    assert hi.comparar_histogramas(h, h, "Correlación") > 0.999
