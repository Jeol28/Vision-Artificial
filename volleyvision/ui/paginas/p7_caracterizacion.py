"""7 · Caracterización."""

import cv2
import gradio as gr
import pandas as pd

from volleyvision.core import calibracion as cal
from volleyvision.core import caracterizacion as ca
from volleyvision.core import histogramas as hi
from volleyvision.core import segmentacion as sg
from volleyvision.ui import componentes as ui
from volleyvision.utils import visualizacion as vis

REGIONES = ("Jugadores (segmentación clásica)", "Candidatos a balón")


def _regiones(img, fuente):
    if fuente == REGIONES[0]:
        return [ca.describir_region(r["contorno"]) for r in sg.segmentar_jugadores_clasico(img)["regiones"]]
    return ca.detectar_balon_clasico(img)


def opciones_regiones(*args):
    *vals, fuente = args
    img, _ = ui.resolver_imagen(*vals)
    n = len(_regiones(img, fuente))
    return gr.Dropdown(choices=[(f"Región {i + 1}", i) for i in range(n)], value=0 if n else None)


def descriptores(*args):
    *vals, fuente, indice = args
    img, _ = ui.resolver_imagen(*vals)
    regiones = _regiones(img, fuente)
    if not regiones:
        return None, "⚠️ No se encontraron regiones.", "", None
    sel = min(int(indice or 0), len(regiones) - 1)
    r = regiones[sel]
    dibujo = sg.dibujar_contornos(img, [x["contorno"] for x in regiones], vis.NARANJA, 1)
    cv2.drawContours(dibujo, [r["contorno"]], -1, vis.VERDE, 2)
    cx, cy = (int(v) for v in r["centroide"])
    cv2.drawMarker(dibujo, (cx, cy), vis.ROJO, cv2.MARKER_CROSS, 14, 2)
    x1, y1, x2, y2 = r["caja"]
    cv2.rectangle(dibujo, (x1, y1), (x2, y2), vis.NEUTRO, 1)
    cv2.drawContours(dibujo, [cv2.convexHull(r["contorno"])], -1, (255, 200, 0), 1)
    tarjetas = ui.metricas({"Área": f"{r['area']:.0f} px²", "Perímetro": f"{r['perimetro']:.1f} px",
                            "Circularidad": f"{r['circularidad']:.3f}", "Excentricidad": f"{r['excentricidad']:.3f}",
                            "Solidez": f"{r['solidez']:.3f}", "Relación alto/ancho": f"{r['relacion_aspecto']:.2f}"})
    hu = "Momentos de Hu (−sign·log₁₀|φ|):\n\n```\n" + " ".join(f"{h:.2f}" for h in r["hu"]) + "\n```"
    return ui.rgb(dibujo), tarjetas, hu, pd.DataFrame(ca.tabla_descriptores(regiones))


def es_balon(*args):
    *vals, circ, exc, sol, col = args
    img, _ = ui.resolver_imagen(*vals)
    cands = ca.detectar_balon_clasico(img, circ_min=float(circ), exc_max=float(exc), solidez_min=float(sol),
                                      frac_color_min=float(col))
    dibujo = img.copy()
    for i, c in enumerate(cands):
        color = vis.VERDE if c["es_balon"] else vis.ROJO
        cv2.drawContours(dibujo, [c["contorno"]], -1, color, 2)
        x1, y1, _, _ = c["caja"]
        cv2.putText(dibujo, str(i + 1), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if cands:
        tabla = pd.DataFrame([{"id": i + 1, "¿balón?": "✅" if c["es_balon"] else "❌",
                               "circularidad": round(c["circularidad"], 3),
                               "excentricidad": round(c["excentricidad"], 3), "solidez": round(c["solidez"], 3),
                               "amarillo": f"{c['frac_color']:.0%}", "área": round(c["area"]),
                               "motivos de rechazo": "; ".join(c["motivos"])} for i, c in enumerate(cands)])
        nota = ""
    else:
        tabla, nota = None, "No hay candidatos de color amarillo o azul en esta imagen."
    return ui.rgb(ca.mascara_candidatos_balon(img)), ui.rgb(dibujo), tabla, nota


def proyeccion(*args):
    *vals, estado = args
    img, nombre = ui.resolver_imagen(*vals)
    esquinas = ui.esquinas_de(estado, nombre)
    if esquinas is None:
        return None, None, None, ("Marca las 4 esquinas de tu media cancha para esta imagen (aquí abajo o en "
                                  "la pestaña de Calibración).")
    hom = cal.homografia_cancha(esquinas)
    cajas = [r["caja"] for r in sg.segmentar_jugadores_clasico(img)["regiones"]]
    proy = ca.proyectar_jugadores(cajas, hom)
    equipos = hi.asignar_equipos(hi.histogramas_de_cajas(img, cajas))[0] if len(cajas) >= 2 else [0] * len(cajas)
    colores = [vis.ROJO if e == 0 else vis.VERDE for e in equipos]
    dibujo = vis.dibujar_cajas(img, [{"caja": c, "color": col} for c, col in zip(cajas, colores)])
    for (x, y), col in zip(proy["puntos_imagen"], colores):
        cv2.circle(dibujo, (int(x), int(y)), 5, col, -1)
    dibujo = vis.dibujar_puntos(dibujo, esquinas, vis.NARANJA, numerar=False, cerrar=True)
    pos = [tuple(p) for p, d in zip(proy["metros"], proy["dentro"]) if d]
    cols = [c for c, d in zip(colores, proy["dentro"]) if d]
    mapa = gr.Image(ui.rgb(vis.dibujar_cancha_cenital(pos, cols)), label=f"Mapa cenital: {len(pos)} jugadores en cancha")
    tabla = pd.DataFrame([{"jugador": i + 1, "equipo": "AB"[e], "x (m)": round(float(p[0]), 2),
                           "y (m)": round(float(p[1]), 2), "en cancha": bool(d)}
                          for i, (p, e, d) in enumerate(zip(proy["metros"], equipos, proy["dentro"]))])
    return ui.rgb(dibujo), mapa, tabla, ""


def construir(estado: gr.State) -> None:
    ui.encabezado("7 · Caracterización", "Descriptores de forma, clasificación \"¿es balón?\" y posiciones en la "
                                         "cancha.")
    ui.en_pocas_palabras("caracterizar es medir cada objeto encontrado (tamaño, qué tan redondo es, su color) "
                         "para decidir qué es: una forma redonda y amarilla es el balón; una alta y angosta, un "
                         "jugador. Al final se ubica a cada jugador en un mapa de la cancha.")
    sel = ui.selector_imagen()
    E = sel.entradas
    with gr.Tabs():
        with gr.Tab("Descriptores"):
            with gr.Row():
                fuente = gr.Radio(list(REGIONES), value=REGIONES[0], label="Regiones")
                indice = gr.Dropdown([], label="Resaltar región")
            with gr.Row():
                with gr.Column(scale=3):
                    dibujo = ui.imagen_salida("Verde: región · cruz roja: centroide · blanco: caja · celeste: "
                                              "envolvente convexa")
                with gr.Column(scale=2):
                    tarjetas = gr.HTML()
                    hu = ui.md()
            tabla = gr.Dataframe(interactive=False)
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
            gr.on(None, opciones_regiones, E + [fuente], indice, trigger_mode="always_last")
            gr.on(None, descriptores, E + [fuente, indice], [dibujo, tarjetas, hu, tabla],
                  trigger_mode="always_last")

        with gr.Tab("¿Es balón?"):
            circ = gr.Slider(0.3, 1.0, 0.78, step=0.01, label="¿Qué tan redondo debe ser? (circularidad mínima)")
            with ui.opciones_avanzadas(), gr.Row():
                exc = gr.Slider(0.1, 1.0, 0.45, step=0.01, label="Excentricidad máxima")
                sol = gr.Slider(0.5, 1.0, 0.85, step=0.01, label="Solidez mínima")
                col = gr.Slider(0.0, 1.0, 0.4, step=0.05, label="Fracción de amarillo mínima")
            with gr.Row():
                candidatos = ui.imagen_salida("Candidatos (HSV + cierre + relleno)")
                clasificados = ui.imagen_salida("Verde = balón · rojo = descartado")
            tabla_b = gr.Dataframe(interactive=False)
            nota_b = ui.md()
            ui.que_esta_pasando(r"""
Clasificador por **reglas interpretables**: una región es balón si cumple todas:
circularidad $\ge C_{min}$, excentricidad $\le e_{max}$, solidez $\ge s_{min}$, al menos una fracción de
píxeles **amarillos** y un área razonable. Las camisetas azules también entran como candidatas (por el
color azul del balón), pero son rectangulares: la excentricidad o el color las descartan. Mueve los
umbrales para ver cómo cambian los aciertos.
""")
            gr.on(None, es_balon, E + [circ, exc, sol, col], [candidatos, clasificados, tabla_b, nota_b],
                  trigger_mode="always_last")

        with gr.Tab("Proyección a la cancha"):
            nota_p = ui.md()
            with gr.Row():
                apoyo = ui.imagen_salida("Punto de apoyo de cada jugador")
                mapa = ui.imagen_salida("Mapa cenital")
            tabla_p = gr.Dataframe(interactive=False)
            with gr.Accordion("Marcar las esquinas de la media cancha", open=False):
                ui.selector_esquinas(sel, estado)
            ui.que_esta_pasando(r"""
Con la homografía $H$ de la pestaña de calibración, cada punto de la imagen se lleva a metros:
$[X, Y, 1]^\top \sim H\,[u, v, 1]^\top$. Se proyecta el **punto de apoyo** (centro inferior de la caja,
donde están los pies) porque $H$ solo es válida para puntos del **plano del piso**. El centroide está a
~1 m de altura y daría posiciones desplazadas hacia el fondo. Los colores son los equipos obtenidos con
la comparación de histogramas de la pestaña 3.
""")
            gr.on(None, proyeccion, E + [estado], [apoyo, mapa, tabla_p, nota_p], trigger_mode="always_last")
