import cv2
import numpy as np
import pytest
from skimage.measure import label, regionprops

from volleyvision.core import calibracion as cal
from volleyvision.core import caracterizacion as ca


def _mascara_circulo(r=40, tam=120):
    m = np.zeros((tam, tam), np.uint8)
    cv2.circle(m, (tam // 2, tam // 2), r, 255, -1)
    return m


def _mascara_elipse(ejes=(50, 15), ang=30, tam=160):
    m = np.zeros((tam, tam), np.uint8)
    cv2.ellipse(m, (tam // 2, tam // 2), ejes, ang, 0, 360, 255, -1)
    return m


def test_circulo_es_circular():
    d = ca.describir_mascara(_mascara_circulo())[0]
    assert d["circularidad"] > 0.85
    assert d["excentricidad"] < 0.1
    assert d["solidez"] > 0.98
    assert d["centroide"] == pytest.approx((60, 60), abs=0.5)
    assert d["area"] == pytest.approx(np.pi * 40 ** 2, rel=0.03)


def test_excentricidad_coincide_con_skimage():
    m = _mascara_elipse()
    d = ca.describir_mascara(m)[0]
    ref = regionprops(label(m > 0))[0]
    assert d["excentricidad"] == pytest.approx(ref.eccentricity, abs=0.02)
    assert d["area"] == pytest.approx(ref.area, rel=0.05)  # el contorno pasa por el centro de los píxeles del borde
    assert d["solidez"] == pytest.approx(ref.solidity, abs=0.03)


def _triangulo(angulo=0.0, escala=1.0, tam=300):
    pts = np.array([[-60, -40], [70, -20], [-10, 80]], float) * escala
    t = np.deg2rad(angulo)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    pts = pts @ R.T + tam / 2
    m = np.zeros((tam, tam), np.uint8)
    cv2.fillPoly(m, [pts.round().astype(np.int32)], 255)
    return m


def test_hu_invariante_a_rotacion_y_escala():
    # Figura asimétrica: en una elipse φ3..φ7 son ~0 y su logaritmo es solo ruido numérico
    a = ca.describir_mascara(_triangulo())[0]["hu"]
    b = ca.describir_mascara(_triangulo(angulo=65))[0]["hu"]
    c = ca.describir_mascara(_triangulo(angulo=-30, escala=0.6))[0]["hu"]
    np.testing.assert_allclose(a[:4], b[:4], rtol=0.05)  # la discretización altera los valores pequeños
    np.testing.assert_allclose(a[:4], c[:4], rtol=0.06)


def test_clasifica_balon_en_escena(escena):
    cands = ca.detectar_balon_clasico(escena.imagen)
    balones = [c for c in cands if c["es_balon"]]
    assert len(balones) == 1
    cx, cy = balones[0]["centroide"]
    assert np.hypot(cx - escena.balon["centro"][0], cy - escena.balon["centro"][1]) < 3
    # las camisetas azules aparecen como candidatos pero se rechazan
    assert len(cands) > 1
    assert all(c["motivos"] for c in cands if not c["es_balon"])


def test_proyeccion_de_jugadores(escena):
    hom = cal.homografia_cancha(escena.esquinas)
    res = ca.proyectar_jugadores([j["caja"] for j in escena.jugadores], hom)
    reales = np.array([j["pos_m"] for j in escena.jugadores])
    assert res["dentro"].all()
    assert np.abs(res["metros"] - reales).max() < 0.2


def test_tabla_descriptores():
    filas = ca.tabla_descriptores(ca.describir_mascara(_mascara_circulo()))
    assert {"id", "area", "circularidad", "Hu1", "Hu7", "cx"} <= set(filas[0])


def test_equipo_propio_por_posicion(escena):
    """Con la homografía, son propios exactamente los jugadores de la media cancha propia (equipo A)."""
    cajas = [j["caja"] for j in escena.jugadores]
    sel = ca.seleccionar_equipo_propio(cajas, hom=cal.homografia_cancha(escena.esquinas))
    assert sel["propio"].tolist() == [j["equipo"] == "A" for j in escena.jugadores]
    assert "homografía" in sel["criterio"]


def test_equipo_propio_sin_esquinas_es_el_mas_cercano(escena):
    """Sin homografía se elige el grupo de camisetas con los pies más abajo (más cerca de la cámara)."""
    cajas = [j["caja"] for j in escena.jugadores]
    equipos = [1 if j["equipo"] == "A" else 0 for j in escena.jugadores]  # etiquetas al revés a propósito
    sel = ca.seleccionar_equipo_propio(cajas, equipos)
    assert sel["propio"].tolist() == [j["equipo"] == "A" for j in escena.jugadores]


def test_zona_propia():
    assert ca.zona_propia(10.0).startswith("ataque") and ca.zona_propia(15.0).startswith("defensa")
    assert ca.seleccionar_equipo_propio([])["propio"].size == 0
