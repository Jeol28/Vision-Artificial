import cv2
import numpy as np

from volleyvision.core import calibracion as cal
from volleyvision.utils import sinteticos


def test_calibracion_recupera_intrinsecos(vistas_tablero):
    res = cal.calibrar_camara(vistas_tablero, sinteticos.PATRON, sinteticos.TAM_CUADRO_MM)
    K = sinteticos.K_VERDADERA
    assert res.error_rms < 0.5
    assert abs(res.K[0, 0] - K[0, 0]) / K[0, 0] < 0.03
    assert abs(res.K[1, 1] - K[1, 1]) / K[1, 1] < 0.03
    assert abs(res.K[0, 2] - K[0, 2]) < 10
    assert abs(res.K[1, 2] - K[1, 2]) < 10
    assert res.dist[0] < 0  # distorsión de barril


def test_proyeccion_pinhole_manual_igual_a_opencv(rng):
    pts = rng.uniform(-1, 1, (20, 3)) + [0, 0, 5]
    rvec = np.array([0.1, -0.2, 0.05])
    R, _ = cv2.Rodrigues(rvec)
    t = np.array([0.2, -0.1, 0.5])
    K = sinteticos.K_VERDADERA
    manual = cal.proyectar_pinhole(pts, K, R, t)
    ocv, _ = cv2.projectPoints(pts, rvec, t, K, None)
    np.testing.assert_allclose(manual, ocv.reshape(-1, 2), atol=1e-6)


def test_dlt_manual_igual_a_opencv(escena):
    metros = np.array([[0, 0], [18, 0], [18, 9], [0, 9]], np.float32)
    H_manual = cal.homografia_dlt(escena.esquinas, metros)
    H_ocv = cv2.getPerspectiveTransform(escena.esquinas, metros)
    np.testing.assert_allclose(H_manual, H_ocv / H_ocv[2, 2], rtol=1e-4, atol=1e-6)


def test_homografia_cancha_lleva_esquinas_a_metros(escena):
    hom = cal.homografia_cancha(escena.esquinas)
    metros = cal.imagen_a_cancha(escena.esquinas, hom)
    np.testing.assert_allclose(metros, [[0, 0], [18, 0], [18, 9], [0, 9]], atol=1e-3)


def test_proyeccion_de_jugadores_a_cancha(escena):
    hom = cal.homografia_cancha(escena.esquinas)
    pies = [((j["caja"][0] + j["caja"][2]) / 2, j["caja"][3]) for j in escena.jugadores]
    metros = cal.imagen_a_cancha(pies, hom)
    reales = np.array([j["pos_m"] for j in escena.jugadores])
    assert np.max(np.linalg.norm(metros - reales, axis=1)) < 0.15
    assert cal.esta_en_cancha(metros, hom).all()


def test_vista_cenital_tamano(escena):
    hom = cal.homografia_cancha(escena.esquinas, px_por_m=20, margen_m=1)
    cen = cal.vista_cenital(escena.imagen, hom)
    assert cen.shape[:2] == (int(11 * 20), int(20 * 20))


def test_correccion_distorsion_endereza_rectas(vistas_tablero):
    res = cal.calibrar_camara(vistas_tablero, sinteticos.PATRON, sinteticos.TAM_CUADRO_MM)
    img = vistas_tablero[0]

    def curvatura(im):
        ok, esq = cal.encontrar_esquinas(im, sinteticos.PATRON)
        assert ok
        filas = esq.reshape(sinteticos.PATRON[1], sinteticos.PATRON[0], 2)
        err = 0
        for f in filas:
            coef = np.polyfit(f[:, 0], f[:, 1], 1)
            err += np.abs(np.polyval(coef, f[:, 0]) - f[:, 1]).max()
        return err

    corregida, _ = cal.corregir_distorsion(img, res.K, res.dist, alfa=1.0)
    assert curvatura(corregida) < curvatura(img)
