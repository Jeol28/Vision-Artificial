import cv2
import pandas as pd
import streamlit as st

from volleyvision.core import calibracion as cal
from volleyvision.core import caracterizacion as ca
from volleyvision.core import histogramas as hi
from volleyvision.core import segmentacion as sg
from volleyvision.ui import componentes as ui
from volleyvision.utils import visualizacion as vis

ui.encabezado("7 · Caracterización", "Descriptores de forma, clasificación \"¿es balón?\" y posiciones en la cancha.")
img, nombre = ui.selector_imagen("car")

tab_desc, tab_balon, tab_mapa = st.tabs(["Descriptores", "¿Es balón?", "Proyección a la cancha"])
jug = sg.segmentar_jugadores_clasico(img)

with tab_desc:
    fuente = st.radio("Regiones", ["Jugadores (segmentación clásica)", "Candidatos a balón"], horizontal=True,
                      key="car_fuente")
    if fuente.startswith("Jugadores"):
        regiones = [ca.describir_region(r["contorno"]) for r in jug["regiones"]]
    else:
        regiones = ca.detectar_balon_clasico(img)
    if not regiones:
        st.warning("No se encontraron regiones.")
    else:
        sel = st.selectbox("Resaltar región", range(len(regiones)), format_func=lambda i: f"Región {i + 1}",
                           key="car_sel")
        r = regiones[sel]
        dibujo = sg.dibujar_contornos(img, [x["contorno"] for x in regiones], vis.NARANJA, 1)
        cv2.drawContours(dibujo, [r["contorno"]], -1, vis.VERDE, 2)
        cx, cy = (int(v) for v in r["centroide"])
        cv2.drawMarker(dibujo, (cx, cy), vis.ROJO, cv2.MARKER_CROSS, 14, 2)
        x1, y1, x2, y2 = r["caja"]
        cv2.rectangle(dibujo, (x1, y1), (x2, y2), vis.NEUTRO, 1)
        hull = cv2.convexHull(r["contorno"])
        cv2.drawContours(dibujo, [hull], -1, (255, 200, 0), 1)
        c1, c2 = st.columns([3, 2])
        with c1:
            ui.mostrar(dibujo, "Verde: región · cruz roja: centroide · blanco: caja · celeste: envolvente convexa")
        with c2:
            m = st.columns(2)
            m[0].metric("Área", f"{r['area']:.0f} px²")
            m[1].metric("Perímetro", f"{r['perimetro']:.1f} px")
            m[0].metric("Circularidad", f"{r['circularidad']:.3f}")
            m[1].metric("Excentricidad", f"{r['excentricidad']:.3f}")
            m[0].metric("Solidez", f"{r['solidez']:.3f}")
            m[1].metric("Relación alto/ancho", f"{r['relacion_aspecto']:.2f}")
            st.write("Momentos de Hu (−sign·log₁₀|φ|):")
            st.code(" ".join(f"{h:.2f}" for h in r["hu"]))
        st.dataframe(pd.DataFrame(ca.tabla_descriptores(regiones)), hide_index=True, width="stretch")
    ui.que_esta_pasando(r"""
Momentos $m_{pq}=\sum x^p y^q$: **área** $m_{00}$, **centroide** $(m_{10}/m_{00},\,m_{01}/m_{00})$.

* **Circularidad** $C = 4\pi A/P^2$ (1 = círculo).
* **Excentricidad** $e=\sqrt{1-\lambda_2/\lambda_1}$, con $\lambda_i$ valores propios de la matriz de
  covarianza $\begin{bmatrix}\mu_{20}&\mu_{11}\\\mu_{11}&\mu_{02}\end{bmatrix}/m_{00}$ (0 = círculo).
* **Solidez** $A/A_{\text{convexa}}$: un jugador con los brazos abiertos tiene solidez baja.
* **Momentos de Hu**: 7 combinaciones de momentos centrales normalizados $\eta_{pq}$, invariantes a
  traslación, escala y rotación. Sirven para comparar formas independientemente de su tamaño.

Código: `core/caracterizacion.py` → `describir_region`.
""")

with tab_balon:
    c1, c2, c3, c4 = st.columns(4)
    circ = c1.slider("Circularidad mínima", 0.3, 1.0, 0.78, 0.01, key="car_circ")
    exc = c2.slider("Excentricidad máxima", 0.1, 1.0, 0.45, 0.01, key="car_exc")
    sol = c3.slider("Solidez mínima", 0.5, 1.0, 0.85, 0.01, key="car_sol")
    col = c4.slider("Fracción de amarillo mínima", 0.0, 1.0, 0.4, 0.05, key="car_col")
    cands = ca.detectar_balon_clasico(img, circ_min=circ, exc_max=exc, solidez_min=sol, frac_color_min=col)
    dibujo = img.copy()
    for i, c in enumerate(cands):
        color = vis.VERDE if c["es_balon"] else vis.ROJO
        cv2.drawContours(dibujo, [c["contorno"]], -1, color, 2)
        x1, y1, _, _ = c["caja"]
        cv2.putText(dibujo, str(i + 1), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    ui.lado_a_lado(ca.mascara_candidatos_balon(img), dibujo, "Candidatos (HSV + cierre + relleno)",
                   "Verde = balón · rojo = descartado")
    if cands:
        st.dataframe(pd.DataFrame([{"id": i + 1, "¿balón?": "✅" if c["es_balon"] else "❌",
                                    "circularidad": round(c["circularidad"], 3),
                                    "excentricidad": round(c["excentricidad"], 3), "solidez": round(c["solidez"], 3),
                                    "amarillo": f"{c['frac_color']:.0%}", "área": round(c["area"]),
                                    "motivos de rechazo": "; ".join(c["motivos"])} for i, c in enumerate(cands)]),
                     hide_index=True, width="stretch")
    else:
        st.info("No hay candidatos de color amarillo o azul en esta imagen.")
    ui.que_esta_pasando(r"""
Clasificador por **reglas interpretables**: una región es balón si cumple todas:
circularidad $\ge C_{min}$, excentricidad $\le e_{max}$, solidez $\ge s_{min}$, al menos una fracción de
píxeles **amarillos** y un área razonable. Las camisetas azules también entran como candidatas (por el
color azul del balón), pero son rectangulares: la excentricidad o el color las descartan. Mueve los
umbrales para ver cómo cambian los aciertos.
""")

with tab_mapa:
    esquinas = ui.esquinas_de(nombre)
    if esquinas is None:
        st.info("Primero marca las 4 esquinas de la cancha para esta imagen.")
        esquinas = ui.selector_esquinas(img, nombre, "car_esq")
    if esquinas is not None:
        hom = cal.homografia_cancha(esquinas)
        cajas = [r["caja"] for r in jug["regiones"]]
        proy = ca.proyectar_jugadores(cajas, hom)
        equipos = hi.asignar_equipos(hi.histogramas_de_cajas(img, cajas))[0] if len(cajas) >= 2 else [0] * len(cajas)
        colores = [vis.ROJO if e == 0 else vis.VERDE for e in equipos]
        dibujo = vis.dibujar_cajas(img, [{"caja": c, "color": col} for c, col in zip(cajas, colores)])
        for (x, y), col in zip(proy["puntos_imagen"], colores):
            cv2.circle(dibujo, (int(x), int(y)), 5, col, -1)
        dibujo = vis.dibujar_puntos(dibujo, esquinas, vis.NARANJA, numerar=False, cerrar=True)
        pos = [tuple(p) for p, d in zip(proy["metros"], proy["dentro"]) if d]
        cols = [c for c, d in zip(colores, proy["dentro"]) if d]
        ui.lado_a_lado(dibujo, vis.dibujar_cancha_cenital(pos, cols), "Punto de apoyo de cada jugador",
                       f"Mapa cenital: {len(pos)} jugadores en cancha")
        st.dataframe(pd.DataFrame([{"jugador": i + 1, "equipo": "AB"[e], "x (m)": round(float(p[0]), 2),
                                    "y (m)": round(float(p[1]), 2), "en cancha": bool(d)}
                                   for i, (p, e, d) in enumerate(zip(proy["metros"], equipos, proy["dentro"]))]),
                     hide_index=True)
    ui.que_esta_pasando(r"""
Con la homografía $H$ de la página de calibración, cada punto de la imagen se lleva a metros:
$[X, Y, 1]^\top \sim H\,[u, v, 1]^\top$. Se proyecta el **punto de apoyo** (centro inferior de la caja,
donde están los pies) porque $H$ solo es válida para puntos del **plano del piso**. El centroide está a
~1 m de altura y daría posiciones desplazadas hacia el fondo. Los colores son los equipos obtenidos con
la comparación de histogramas de la página 3.
""")
