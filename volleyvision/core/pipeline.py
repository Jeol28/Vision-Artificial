"""Pipeline completo de VolleyVision.

Encadena los temas del curso sobre una imagen o un frame de video:

1. Corrección de distorsión (calibración)
2. Mejora de histograma (CLAHE)
3. Filtrado (bilateral + Canny + Hough para las líneas)
4. Umbralización + morfología (máscaras del balón y de los jugadores)
5. Segmentación clásica y transformer
6. Caracterización (descriptores, balón y equipos)
7. Mapa cenital de las posiciones (homografía)

Cada paso devuelve un :class:`Paso` con sus imágenes intermedias, datos y tiempo,
para que la UI pueda mostrar el resultado de cada etapa.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from volleyvision.core import calibracion as cal
from volleyvision.core import caracterizacion as ca
from volleyvision.core import filtros as fi
from volleyvision.core import histogramas as hi
from volleyvision.core import segmentacion as sg
from volleyvision.core import transformers_models as tm
from volleyvision.utils import visualizacion as vis


@dataclass
class ConfigPipeline:
    """Parámetros del pipeline (todos con valores por defecto razonables)."""

    calibracion: cal.ResultadoCalibracion | None = None
    esquinas: np.ndarray | None = None          # 4 esquinas de la cancha en la imagen de entrada
    clip_limit: float = 2.0
    baldosas: int = 8
    d_bilateral: int = 7
    canny_bajo: int = 50
    canny_alto: int = 150
    k_kmeans: int = 5
    tam_cierre: int = 9
    area_min: int = 150
    umbral_deteccion: float = 0.5
    metrica_histograma: str = "Bhattacharyya"


@dataclass
class Paso:
    nombre: str
    imagenes: dict[str, np.ndarray]
    datos: dict = field(default_factory=dict)
    tiempo_ms: float = 0.0
    nota: str = ""


def _cronometro():
    t0 = time.perf_counter()
    return lambda: (time.perf_counter() - t0) * 1e3


def ejecutar_pipeline(img: np.ndarray, config: ConfigPipeline | None = None,
                      detector: tm.ModeloCargado | None = None,
                      segformer: tm.ModeloCargado | None = None) -> list[Paso]:
    """Ejecuta el pipeline completo.

    Args:
        img: imagen BGR.
        config: parámetros; si no hay calibración o esquinas, esos pasos se omiten con una nota.
        detector: DETR/RT-DETR ya cargado (opcional). Sin él solo se usa la vía clásica.
        segformer: SegFormer ya cargado (opcional).
    """
    cfg = config or ConfigPipeline()
    pasos: list[Paso] = []
    esquinas = None if cfg.esquinas is None else np.asarray(cfg.esquinas, np.float32).reshape(4, 2)

    # 1. Corrección de distorsión --------------------------------------------
    t = _cronometro()
    if cfg.calibracion is not None:
        corregida, K_nueva = cal.corregir_distorsion(img, cfg.calibracion.K, cfg.calibracion.dist, alfa=0.0)
        if esquinas is not None:  # las esquinas se marcaron en la imagen original: se corrigen también
            esquinas = cv2.undistortPoints(esquinas.reshape(-1, 1, 2), cfg.calibracion.K, cfg.calibracion.dist,
                                           P=K_nueva).reshape(4, 2)
        nota = "Distorsión de lente corregida con la calibración cargada."
    else:
        corregida = img.copy()
        nota = "Sin calibración: se usa la imagen tal cual (calibra en la página de Calibración)."
    pasos.append(Paso("1. Corrección de distorsión", {"Entrada": img, "Corregida": corregida},
                      tiempo_ms=t(), nota=nota))

    # 2. Histograma -----------------------------------------------------------
    t = _cronometro()
    mejorada = hi.aplicar_clahe(corregida, cfg.clip_limit, cfg.baldosas)
    pasos.append(Paso("2. Mejora de histograma (CLAHE)", {"Antes": corregida, "CLAHE": mejorada}, {
        "Contraste RMS antes": hi.contraste_rms(corregida), "Contraste RMS después": hi.contraste_rms(mejorada),
        "Desv. brillo entre bloques antes": hi.uniformidad_iluminacion(corregida),
        "Desv. brillo entre bloques después": hi.uniformidad_iluminacion(mejorada)}, t()))

    # 3. Filtrado -------------------------------------------------------------
    t = _cronometro()
    lineas = fi.resaltar_lineas_cancha(mejorada, cfg.d_bilateral, cfg.canny_bajo, cfg.canny_alto)
    pasos.append(Paso("3. Filtrado y bordes", {"Bilateral": lineas["suavizada"], "Canny": lineas["bordes"],
                                               "Líneas (Hough)": lineas["lineas"]},
                      {"Segmentos de línea": len(lineas["segmentos"])}, t()))
    # Para segmentar por color se usa la imagen corregida (CLAHE altera los tonos
    # ligeramente); el bilateral sí ayuda a homogeneizar el piso.
    base = fi.filtro_bilateral(corregida, cfg.d_bilateral, 40, 40)

    # 4. Umbralización + morfología ------------------------------------------
    t = _cronometro()
    cand_balon = ca.mascara_candidatos_balon(base)
    candidatos = ca.detectar_balon_clasico(base)
    balones = [c for c in candidatos if c["es_balon"]]
    excluir = np.zeros(img.shape[:2], np.uint8)
    for b in balones:
        cv2.drawContours(excluir, [b["contorno"]], -1, 255, -1)
    excluir = cv2.dilate(excluir, np.ones((7, 7), np.uint8))
    jug = sg.segmentar_jugadores_clasico(base, k=cfg.k_kmeans, tam_cierre=cfg.tam_cierre,
                                         area_min=cfg.area_min, mascara_excluir=excluir)
    pasos.append(Paso("4. Umbralización + morfología", {
        "Distancia al fondo": jug["kmeans"]["distancia"], "Otsu (cruda)": jug["mascara_cruda"],
        "Apertura + cierre": jug["mascara"], "Candidatos a balón (HSV + cierre)": cand_balon},
        {"T de Otsu": jug["kmeans"]["T_otsu"]}, t()))

    # 5. Segmentación ---------------------------------------------------------
    t = _cronometro()
    imgs5 = {"Clásica (contornos)": sg.dibujar_contornos(corregida, [r["contorno"] for r in jug["regiones"]],
                                                          con_caja=True)}
    datos5 = {"Jugadores (clásico)": len(jug["regiones"])}
    personas_tr, balon_tr = [], []
    if detector is not None:
        det = tm.detectar_objetos(corregida, detector, cfg.umbral_deteccion)
        personas_tr = [d for d in det["detecciones"] if d["etiqueta"] == "person"]
        balon_tr = [d for d in det["detecciones"] if d["etiqueta"] == "sports ball"]
        imgs5[f"Transformer ({detector.id_hf})"] = vis.dibujar_cajas(corregida, tm.cajas_para_dibujar(
            det["detecciones"]))
        datos5.update({"Personas (transformer)": len(personas_tr), "Balones (transformer)": len(balon_tr),
                       "Tiempo detector (ms)": det["tiempo_ms"]})
    if segformer is not None:
        sem = tm.segmentar_semantica(corregida, segformer)
        m_pers = tm.mascara_de_clase(sem["etiquetas"], sem["id2label"], "person")
        imgs5["SegFormer: personas"] = vis.superponer_mascara(corregida, m_pers, vis.NARANJA, 0.55)
        datos5["IoU máscara clásica vs SegFormer"] = sg.iou_mascaras(jug["mascara"], m_pers)
        datos5["Tiempo SegFormer (ms)"] = sem["tiempo_ms"]
    pasos.append(Paso("5. Segmentación (clásica y transformer)", imgs5, datos5, t()))

    # 6. Caracterización ------------------------------------------------------
    t = _cronometro()
    usar_tr = detector is not None and len(personas_tr) > 0
    cajas = [d["caja"] for d in personas_tr] if usar_tr else [r["caja"] for r in jug["regiones"]]
    descriptores = [ca.describir_region(r["contorno"]) for r in jug["regiones"]]
    equipos, _ = hi.asignar_equipos(hi.histogramas_de_cajas(corregida, cajas), cfg.metrica_histograma) \
        if len(cajas) >= 2 else ([0] * len(cajas), None)
    colores_eq = _colores_equipos(corregida, cajas, equipos)
    anotada = corregida.copy()
    for caja, e in zip(cajas, equipos):
        anotada = vis.dibujar_cajas(anotada, [{"caja": caja, "etiqueta": f"Eq. {'AB'[e]}", "color": colores_eq[e]}])
    for c in candidatos:
        color = vis.VERDE if c["es_balon"] else vis.ROJO
        cv2.drawContours(anotada, [c["contorno"]], -1, color, 2)
    pasos.append(Paso("6. Caracterización", {"Equipos y balón": anotada}, {
        "tabla_jugadores": ca.tabla_descriptores(descriptores),
        "tabla_balon": [{"id": i + 1, "es_balón": c["es_balon"], "circularidad": round(c["circularidad"], 3),
                         "excentricidad": round(c["excentricidad"], 3), "solidez": round(c["solidez"], 3),
                         "color_balón": round(c["frac_color"], 3),
                         "motivos": "; ".join(c["motivos"]) or "cumple todas las reglas"}
                        for i, c in enumerate(candidatos)],
        "Fuente de jugadores": "transformer" if usar_tr else "clásico",
        "Jugadores por equipo": {"A": equipos.count(0), "B": equipos.count(1)}}, t(),
        "Verde = balón, rojo = candidato descartado; cajas coloreadas con el color medio del uniforme."))

    # 7. Mapa cenital ---------------------------------------------------------
    t = _cronometro()
    if esquinas is not None:
        hom = cal.homografia_cancha(esquinas)
        proy = ca.proyectar_jugadores(cajas, hom)
        dentro = proy["dentro"]
        pos = [tuple(p) for p, d in zip(proy["metros"], dentro) if d]
        cols = [colores_eq[e] for e, d in zip(equipos, dentro) if d]
        mapa = vis.dibujar_cancha_cenital(pos, cols)
        rect = cal.vista_cenital(corregida, hom)
        pasos.append(Paso("7. Mapa cenital", {"Vista cenital (warpPerspective)": rect, "Mapa de posiciones": mapa}, {
            "posiciones_m": [{"jugador": i + 1, "equipo": "AB"[equipos[i]], "x (m)": round(float(p[0]), 2),
                              "y (m)": round(float(p[1]), 2), "en cancha": bool(dentro[i])}
                             for i, p in enumerate(proy["metros"])]}, t(),
            "Se proyecta el punto de apoyo (pies) porque la homografía solo vale para el plano del piso."))
    else:
        pasos.append(Paso("7. Mapa cenital", {}, {}, t(),
                          "Marca las 4 esquinas de la cancha para generar el mapa cenital."))
    return pasos


def _colores_equipos(img, cajas, equipos) -> dict[int, tuple[int, int, int]]:
    """Color representativo (mediana del torso) de cada equipo, aclarado para que se vea en el mapa."""
    colores = {0: vis.ROJO, 1: vis.VERDE}
    for e in (0, 1):
        muestras = [hi.color_dominante(img, c) for c, q in zip(cajas, equipos) if q == e]
        if muestras:
            bgr = np.median(np.array(muestras), axis=0).astype(np.uint8)
            hsv = cv2.cvtColor(bgr[None, None], cv2.COLOR_BGR2HSV)[0, 0]
            hsv[2] = max(int(hsv[2]), 170)
            colores[e] = tuple(int(v) for v in cv2.cvtColor(hsv[None, None], cv2.COLOR_HSV2BGR)[0, 0])
    return colores
