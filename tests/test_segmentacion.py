import cv2
import numpy as np

from volleyvision.core import segmentacion as sg


def _dos_circulos_que_se_tocan():
    m = np.zeros((120, 200), np.uint8)
    cv2.circle(m, (70, 60), 35, 255, -1)
    cv2.circle(m, (128, 60), 35, 255, -1)
    return m


def test_contornos_cuenta_objetos():
    m = np.zeros((100, 100), np.uint8)
    m[10:30, 10:30] = 255
    m[50:90, 50:70] = 255
    m[95:97, 95:97] = 255  # ruido
    cs = sg.encontrar_contornos(m, area_min=10)
    assert len(cs) == 2
    assert cv2.contourArea(cs[0]) > cv2.contourArea(cs[1])


def test_watershed_separa_objetos_que_se_tocan():
    m = _dos_circulos_que_se_tocan()
    assert len(sg.encontrar_contornos(m)) == 1           # contornos: una sola mancha
    img = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    res = sg.watershed_distancia(img, m, frac_umbral=0.7)
    assert res["n_objetos"] == 2                           # watershed: dos objetos
    assert (res["etiquetas"] == -1).any()


def test_kmeans_recupera_colores():
    img = np.zeros((60, 90, 3), np.uint8)
    img[:, :30] = (0, 0, 255)
    img[:, 30:60] = (0, 255, 0)
    img[:, 60:] = (255, 0, 0)
    km = sg.kmeans_color(img, k=3, espacio="BGR")
    assert len(np.unique(km["etiquetas"])) == 3
    np.testing.assert_allclose(sorted(km["proporciones"]), [1 / 3] * 3, atol=1e-6)
    assert np.abs(km["cuantizada"].astype(int) - img).max() <= 2


def test_jugadores_clasico_en_escena(escena):
    res = sg.segmentar_jugadores_clasico(escena.imagen)
    reales = [j["caja"] for j in escena.jugadores]
    ev = sg.emparejar_cajas([r["caja"] for r in res["regiones"]], reales, umbral_iou=0.3)
    assert ev["VP"] >= 9
    assert ev["FP"] <= 3


def test_iou_y_emparejamiento():
    assert sg.iou_cajas((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert sg.iou_cajas((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(sg.iou_cajas((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9
    ev = sg.emparejar_cajas([(0, 0, 10, 10), (50, 50, 60, 60)], [(1, 1, 10, 10)])
    assert (ev["VP"], ev["FP"], ev["FN"]) == (1, 1, 0)
    assert ev["correctas"] == [True, False]
    a = np.zeros((10, 10), np.uint8)
    a[:5] = 1
    assert sg.iou_mascaras(a, a) == 1.0
