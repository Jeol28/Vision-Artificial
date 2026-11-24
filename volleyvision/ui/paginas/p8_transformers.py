"""8 · Transformers: detección y comparación clásico vs. transformer."""

import gradio as gr
import pandas as pd

from volleyvision.core import transformers_models as tm
from volleyvision.ui import componentes as ui
from volleyvision.ui import modelos
from volleyvision.utils import visualizacion as vis

MUESTRA_REAL = "voleibol_03.jpg"
CLASICO = "Clásico (k-means + morfología)"


def deteccion(*args):
    *vals, nombre_det, umbral = args
    img, _ = ui.resolver_imagen(*vals)
    res = modelos.detectar(img, nombre_det, float(umbral))
    resumen = tm.resumen_detecciones(res)
    tarjetas = ui.metricas({"Personas": resumen["personas"], "Balones": resumen["balones"],
                            "Inferencia (red)": f"{res['tiempo_ms']:.0f} ms",
                            "Total (con pre/posproceso)": f"{res['tiempo_total_ms']:.0f} ms"})
    tabla = pd.DataFrame([{"clase": d["etiqueta"], "confianza": round(d["puntaje"], 3),
                           "caja (x1, y1, x2, y2)": str(tuple(round(v) for v in d["caja"]))}
                          for d in res["detecciones"]])
    dibujo = vis.dibujar_cajas(img, tm.cajas_para_dibujar(res["detecciones"]))
    return ui.rgb(dibujo), tarjetas, modelos.info(modelos.detector(nombre_det)), tabla


def comparacion(*args):
    """Clásico y cada detector elegido sobre la misma imagen, con tiempos y aciertos."""
    *vals, umbral, elegidos = args
    img, nombre = ui.resolver_imagen(*vals)
    comp = modelos.comparar(img, list(elegidos or []), float(umbral), ui.verdad_muestra(nombre))
    imagenes, filas = {}, []
    for metodo, c in comp.items():
        dets = c["resultado"]["detecciones"]
        ev = c["evaluacion"]
        if ev is not None:  # verde = acierto, rojo = falso positivo (solo personas)
            personas = [d for d in dets if d["etiqueta"] == "person"]
            otras = [d for d in dets if d["etiqueta"] != "person"]
            cajas = tm.cajas_para_dibujar(personas, ev["correctas"]) + tm.cajas_para_dibujar(otras)
        else:
            cajas = tm.cajas_para_dibujar(dets)
        r = c["resumen"]
        imagenes[f"{metodo} · {r['tiempo_ms']:.0f} ms · personas {r['personas']} · balones {r['balones']}"] = \
            vis.dibujar_cajas(img, cajas)
        fila = {"método": metodo, "tiempo (ms)": round(r["tiempo_ms"], 1), "personas": r["personas"],
                "balones": r["balones"]}
        if ev is not None:
            fila.update({"VP": ev["VP"], "FP": ev["FP"], "FN": ev["FN"], "precisión": round(ev["precision"], 2),
                         "exhaustividad": round(ev["exhaustividad"], 2)})
        filas.append(fila)
    fig = vis.grafica_barras([f["método"].split(" (")[0] for f in filas], [f["tiempo (ms)"] for f in filas],
                             "Tiempo de inferencia", [vis.HEX["neutro"], vis.HEX["naranja"], "#4C8DFF"][:len(filas)],
                             "ms")
    nota = ""
    if next(iter(comp.values()))["evaluacion"] is not None:
        nota = ("<span class='vv-nota'>Escena sintética con verdad conocida: verde = persona bien detectada, "
                "rojo = falso positivo.</span>")
    return ui.galeria(imagenes), pd.DataFrame(filas), ui.figura(fig), nota


def construir(estado: gr.State) -> None:
    ui.encabezado("8 · Transformers", "Detección de personas y balón con RT-DETR / DETR y comparación con el "
                                      "método clásico.")
    ui.en_pocas_palabras("RT-DETR y DETR son modelos de inteligencia artificial ya entrenados con miles de fotos: "
                         "encuentran personas y balones sin que les digamos de qué color son. Aquí se comparan con "
                         "el método clásico, en aciertos y en tiempo.")
    sel = ui.selector_imagen(por_defecto=MUESTRA_REAL)
    E = sel.entradas
    if modelos.EN_ZEROGPU:
        ui.md("<span class='vv-nota'>Los modelos corren en una GPU de ZeroGPU que se asigna en cada ejecución "
              "(la primera puede tardar unos segundos más en arrancar).</span>")
    with gr.Tabs():
        with gr.Tab("Detección"):
            with gr.Row():
                nombre_det = gr.Dropdown(list(tm.MODELOS_DETECCION), value=next(iter(tm.MODELOS_DETECCION)),
                                         label="Detector transformer")
                umbral = gr.Slider(0.05, 0.95, 0.5, step=0.05, label="Umbral de confianza")
            boton = gr.Button("Detectar", variant="primary")
            with gr.Row():
                with gr.Column(scale=3):
                    dibujo = ui.imagen_salida("Naranja = persona · amarillo = balón (sports ball)")
                with gr.Column(scale=1):
                    tarjetas = gr.HTML()
                    info = ui.md()
            tabla = gr.Dataframe(interactive=False)
            ui.que_esta_pasando(r"""
**DETR** convierte la detección en predicción de un **conjunto**: CNN (ResNet-50) → codificador
transformer con auto-atención $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ (cada región "mira" toda la
imagen) → decodificador con 100 *object queries* que devuelven directamente (clase, caja). No usa
anclas ni NMS: en el entrenamiento el emparejamiento húngaro asigna cada query a un objeto.

**RT-DETR** mantiene la idea pero con un codificador híbrido eficiente y selección de queries por IoU:
es más rápido y más preciso. Ambos están entrenados en COCO, que tiene `person` y `sports ball`.
Se usan **solo en inferencia** con los pesos públicos.
""")
            boton.click(deteccion, E + [nombre_det, umbral], [dibujo, tarjetas, info, tabla])

        with gr.Tab("Comparación clásico vs. transformer"):
            with gr.Row():
                umbral_c = gr.Slider(0.05, 0.95, 0.5, step=0.05, label="Umbral de confianza (transformers)")
                elegidos = gr.CheckboxGroup(list(tm.MODELOS_DETECCION), value=list(tm.MODELOS_DETECCION),
                                            label="Detectores")
            boton_c = gr.Button("Comparar", variant="primary")
            gal = ui.nueva_galeria(3)
            with gr.Row():
                with gr.Column(scale=3):
                    tabla_c = gr.Dataframe(interactive=False)
                    nota_c = ui.md()
                with gr.Column(scale=2):
                    tiempos = gr.Plot(show_label=False)
            ui.que_esta_pasando(r"""
Misma imagen, tres métodos:

* **Clásico**: k-means + Otsu + morfología + reglas de forma/color. No necesita entrenamiento y es
  explicable, pero depende de colores y de que el fondo sea simple: en fotos reales confunde al público
  con jugadores y pierde a los superpuestos.
* **RT-DETR** y **DETR**: aprendieron de COCO qué es una persona y un balón, así que generalizan a fotos
  reales. RT-DETR suele ser bastante más rápido que DETR.

El tiempo medido es solo la red (sin pre/posproceso), después de una inferencia de calentamiento.
En la escena sintética, los jugadores son figuras geométricas y los transformers detectan pocos: es un
ejemplo de **cambio de dominio**.
""")
            boton_c.click(comparacion, E + [umbral_c, elegidos], [gal, tabla_c, tiempos, nota_c])
