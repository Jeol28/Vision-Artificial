"""6 · Segmentación."""

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from volleyvision.core import segmentacion as sg
from volleyvision.core import transformers_models as tm
from volleyvision.ui import componentes as ui
from volleyvision.ui import modelos
from volleyvision.utils import visualizacion as vis

DEMOS_WS = ("Jugadores de la imagen", "Demo: dos balones que se tocan")
PROMPTS_SAM = ("Punto (clic sobre la imagen)", "Caja (de la detección clásica)")


def kmeans(*args):
    *vals, k, espacio = args
    img, _ = ui.resolver_imagen(*vals)
    k = int(k)
    km = sg.kmeans_color(img, k, espacio)
    colores = [("#%02x%02x%02x" % tuple(int(v) for v in c[::-1])) for c in km["centros_bgr"]]
    fig = vis.grafica_barras([str(i) for i in range(k)], list(km["proporciones"]), "Fracción del área por grupo",
                             colores)
    return ui.rgb(img), gr.Image(ui.rgb(km["cuantizada"]), label=f"K-means (k = {k}, {espacio})"), ui.figura(fig)


def contornos(*args):
    *vals, area_min = args
    img, _ = ui.resolver_imagen(*vals)
    jug = sg.segmentar_jugadores_clasico(img)
    cs = sg.encontrar_contornos(jug["mascara"], int(area_min))
    return ui.rgb(jug["mascara"]), gr.Image(ui.rgb(sg.dibujar_contornos(img, cs, con_caja=True)),
                                            label=f"{len(cs)} contornos")


def watershed(*args):
    *vals, demo, frac = args
    if demo == DEMOS_WS[1]:
        m = np.zeros((240, 360), np.uint8)
        cv2.circle(m, (130, 120), 70, 255, -1)
        cv2.circle(m, (255, 120), 60, 255, -1)
        base = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    else:
        img, _ = ui.resolver_imagen(*vals)
        m, base = sg.segmentar_jugadores_clasico(img)["mascara"], img
    ws = sg.watershed_distancia(base, m, float(frac))
    gal = ui.galeria({"Máscara": m, "Transformada de distancia": ws["distancia"],
                      "Frente seguro (marcadores)": ws["frente_seguro"], "Zona desconocida": ws["desconocido"],
                      "Etiquetas": vis.colorear_etiquetas(np.clip(ws["etiquetas"] - 1, 0, None)),
                      f"Watershed: {ws['n_objetos']} objetos": ws["resultado"]})
    texto = (f"Contornos encontrarían **{len(sg.encontrar_contornos(m))}** objeto(s); "
             f"watershed encuentra **{ws['n_objetos']}**.")
    return gal, texto


def jugadores(*args):
    *vals, kk, cierre, usar_ws = args
    img, nombre = ui.resolver_imagen(*vals)
    r = sg.segmentar_jugadores_clasico(img, int(kk), tam_cierre=int(cierre), usar_watershed=bool(usar_ws))
    verdad = ui.verdad_muestra(nombre)
    cajas = [x["caja"] for x in r["regiones"]]
    texto = ""
    if verdad is not None:
        ev = sg.emparejar_cajas(cajas, verdad, 0.3)
        dibujo = vis.dibujar_cajas(img, [{"caja": c, "color": vis.VERDE if ok else vis.ROJO}
                                         for c, ok in zip(cajas, ev["correctas"])])
        texto = (f"Verdaderos positivos **{ev['VP']}** · falsos positivos **{ev['FP']}** · "
                 f"no detectados **{ev['FN']}** (verde = acierto, rojo = error)")
    else:
        dibujo = vis.dibujar_cajas(img, [{"caja": c} for c in cajas])
    gal = ui.galeria({f"Distancia al fondo (T Otsu = {r['kmeans']['T_otsu']})": r["kmeans"]["distancia"],
                      "Máscara limpia": r["mascara"], f"{len(cajas)} jugadores": dibujo})
    return texto, gal


def segformer(*vals):
    img, _ = ui.resolver_imagen(*vals)
    sem = modelos.segmentar_semantica(img)
    pers = tm.mascara_de_clase(sem["etiquetas"], sem["id2label"], "person")
    clasica = sg.segmentar_jugadores_clasico(img)["mascara"]
    gal = ui.galeria({"Mapa semántico (ADE20K)": vis.colorear_etiquetas(sem["etiquetas"] + 1),
                      "Clase 'person'": vis.superponer_mascara(img, pers, vis.NARANJA, 0.55),
                      "Máscara clásica": clasica})
    tarjetas = ui.metricas({"Tiempo de inferencia": f"{sem['tiempo_ms']:.0f} ms",
                            "IoU clásica vs SegFormer (personas)": f"{sg.iou_mascaras(clasica, pers):.2f}"})
    tabla = pd.DataFrame([{"clase": n, "área": f"{f:.1%}"} for n, f in sem["clases_presentes"][:10]])
    return gal, tarjetas, tabla, modelos.info(modelos.segformer())


def cajas_clasicas(*vals):
    """Opciones del desplegable de cajas para el prompt de SAM."""
    img, _ = ui.resolver_imagen(*vals)
    cajas = [x["caja"] for x in sg.segmentar_jugadores_clasico(img)["regiones"]]
    opciones = [(f"Jugador {i + 1}: {tuple(int(v) for v in c)}", i) for i, c in enumerate(cajas)]
    return ui.rgb(img), gr.Dropdown(choices=opciones, value=0 if opciones else None)


def _sam(img, nombre_sam, punto=None, caja=None):
    r = modelos.segmentar_con_sam(img, nombre_sam, punto=punto, caja=caja)
    base = vis.dibujar_puntos(img, [punto], vis.NARANJA, numerar=False) if punto else \
        vis.dibujar_cajas(img, [{"caja": caja}])
    gal = ui.galeria({f"Máscara {i + 1} (IoU est. {p:.2f})": vis.superponer_mascara(base, m, vis.VERDE, 0.5)
                      for i, (m, p) in enumerate(zip(r["mascaras"], r["puntajes"]))})
    return gal, ui.metricas({"Tiempo de inferencia": f"{r['tiempo_ms']:.0f} ms"}), \
        modelos.info(modelos.sam(nombre_sam))


def sam_punto(fuente, muestra, subida, video, video_subido, frame, nombre_sam, prompt, evt: gr.SelectData):
    # Gradio solo inyecta ``gr.SelectData`` en parámetros con nombre (no en ``*args``).
    if prompt != PROMPTS_SAM[0]:
        raise gr.Error("Elige el prompt «Punto» para segmentar con un clic.")
    img, _ = ui.resolver_imagen(fuente, muestra, subida, video, video_subido, frame)
    return _sam(img, nombre_sam, punto=(float(evt.index[0]), float(evt.index[1])))


def sam_caja(*args):
    *vals, nombre_sam, indice = args
    img, _ = ui.resolver_imagen(*vals)
    cajas = [x["caja"] for x in sg.segmentar_jugadores_clasico(img)["regiones"]]
    if indice is None or not cajas:
        raise gr.Error("La segmentación clásica no encontró cajas.")
    return _sam(img, nombre_sam, caja=cajas[int(indice)])


def construir(estado: gr.State) -> None:
    ui.encabezado("6 · Segmentación", "Métodos clásicos (contornos, watershed, k-means) y transformer "
                                      "(SegFormer, SAM).")
    ui.en_pocas_palabras("segmentar es dividir la imagen en partes con sentido: piso, jugadores, balón. Los "
                         "métodos clásicos agrupan por color y forma; los transformers (SegFormer y SAM) son "
                         "modelos de inteligencia artificial que ya aprendieron cómo se ve una persona.")
    sel = ui.selector_imagen()
    E = sel.entradas
    with gr.Tabs():
        with gr.Tab("K-means"):
            with gr.Row():
                k = gr.Slider(2, 12, 6, step=1, label="k (número de colores)")
                espacio = gr.Radio(list(sg.ESPACIOS_KMEANS), value=list(sg.ESPACIOS_KMEANS)[0], label="Espacio")
            with gr.Row():
                original = ui.imagen_salida("Original")
                cuantizada = ui.imagen_salida("K-means")
            barras = gr.Plot(show_label=False)
            ui.que_esta_pasando(r"""
K-means minimiza $J=\sum_j\sum_{x_i\in C_j}\lVert x_i-\mu_j\rVert^2$ alternando: asignar cada píxel al
centro más cercano y recalcular los centros. En **Lab** la distancia euclidiana se parece a la diferencia
de color percibida. Los grupos grandes son el fondo (cancha, zona libre, gradas); los jugadores y el
balón quedan en grupos pequeños o mezclados. Código: `core/segmentacion.py` → `kmeans_color`.
""")
            gr.on(None, kmeans, E + [k, espacio], [original, cuantizada, barras], trigger_mode="always_last")

        with gr.Tab("Contornos"):
            area_min = gr.Slider(0, 2000, 150, step=10, label="Área mínima del contorno (px)")
            with gr.Row():
                mascara = ui.imagen_salida("Máscara de entrada")
                dibujados = ui.imagen_salida("Contornos")
            ui.que_esta_pasando(r"""
`cv2.findContours` (algoritmo de Suzuki–Abe) recorre la frontera de cada componente conexa de una
máscara binaria. Con `RETR_EXTERNAL` se toman solo los contornos exteriores y con `CHAIN_APPROX_SIMPLE`
se guardan solo los vértices. Si dos jugadores se tocan, salen **un solo** contorno: para eso está watershed.
""")
            gr.on(None, contornos, E + [area_min], [mascara, dibujados], trigger_mode="always_last")

        with gr.Tab("Watershed"):
            with gr.Row():
                demo = gr.Radio(list(DEMOS_WS), value=DEMOS_WS[0], label="Máscara")
                frac = gr.Slider(0.1, 0.95, 0.6, step=0.05,
                                 label="Umbral del frente seguro (fracción del máximo de la distancia)")
            gal_ws = ui.nueva_galeria(3)
            texto_ws = ui.md()
            ui.que_esta_pasando(r"""
1. **Transformada de distancia** $D(p)=\min_{q\notin A}\lVert p-q\rVert$: es máxima en el centro de cada objeto.
2. **Marcadores**: píxeles con $D \ge \alpha\cdot\max D$ de su componente (frente seguro).
3. **Desconocido** = fondo seguro (dilatación) − frente seguro.
4. `cv2.watershed` "inunda" el relieve desde cada marcador; donde se encuentran dos cuencas traza la
   frontera (en naranja). Separa objetos que se tocan.
""")
            gr.on(None, watershed, E + [demo, frac], [gal_ws, texto_ws], trigger_mode="always_last")

        with gr.Tab("Jugadores (clásico)"):
            usar_ws = gr.Checkbox(False, label="Separar jugadores pegados (watershed)")
            with ui.opciones_avanzadas(), gr.Row():
                kk = gr.Slider(3, 8, 6, step=1, label="k del modelo de fondo")
                cierre = gr.Slider(3, 21, 9, step=2, label="Cierre vertical (px)")
            texto_j = ui.md()
            gal_j = ui.nueva_galeria(3)
            ui.que_esta_pasando(r"""
Cadena clásica completa (sin aprendizaje):
k-means encuentra los colores del **fondo** → distancia de cada píxel al color de fondo más cercano →
**Otsu** → quitar líneas blancas → apertura + **cierre vertical** → contornos → filtro de forma
(alto/ancho ≥ 1.3). Funciona en la escena sintética, pero en fotos reales falla con el público, las
sombras y los jugadores superpuestos. Es la motivación para los transformers.
""")
            gr.on(None, jugadores, E + [kk, cierre, usar_ws], [texto_j, gal_j], trigger_mode="always_last")

        with gr.Tab("SegFormer"):
            boton_seg = gr.Button(f"Ejecutar SegFormer ({tm.MODELO_SEGFORMER})", variant="primary")
            gal_seg = ui.nueva_galeria(3)
            with gr.Row():
                tarjetas_seg = gr.HTML()
                tabla_seg = gr.Dataframe(headers=["clase", "área"], interactive=False)
            info_seg = ui.md()
            ui.que_esta_pasando(r"""
**SegFormer** asigna una de las 150 clases de ADE20K a **cada píxel**. Su codificador es un transformer
jerárquico (*Mix Transformer*) que produce mapas a 1/4, 1/8, 1/16 y 1/32 de la resolución con
auto-atención eficiente $\mathrm{softmax}(QK^\top/\sqrt{d})V$ (las claves se reducen espacialmente), y un
decodificador MLP fusiona las escalas. No necesita codificación posicional.
""")
            boton_seg.click(segformer, E, [gal_seg, tarjetas_seg, tabla_seg, info_seg])

        with gr.Tab("SAM / SlimSAM"):
            with gr.Row():
                with gr.Column(scale=1):
                    nombre_sam = gr.Dropdown(list(tm.MODELOS_SAM), value=next(iter(tm.MODELOS_SAM)),
                                             label="Modelo")
                    prompt = gr.Radio(list(PROMPTS_SAM), value=PROMPTS_SAM[0], label="Prompt")
                    caja = gr.Dropdown([], label="Caja", visible=False)
                    boton_caja = gr.Button("Segmentar la caja", visible=False)
                    ui.md("<span class='vv-nota'>Con el prompt «Punto», haz clic sobre el balón o un "
                          "jugador.</span>")
                with gr.Column(scale=2):
                    imagen_sam = gr.Image(label="Haz clic para segmentar", interactive=False, format="png")
            gal_sam = ui.nueva_galeria(3)
            tarjetas_sam = gr.HTML()
            info_sam = ui.md()
            ui.que_esta_pasando(r"""
**SAM** (*Segment Anything*) segmenta "lo que se le señale": un codificador ViT procesa la imagen, un
codificador de *prompts* convierte el punto o la caja en tokens, y un decodificador ligero propone
**3 máscaras** (parte, objeto, todo) con su IoU estimado. **SlimSAM** es SAM podado y destilado
(~10 % de los parámetros), viable en CPU. A diferencia de SegFormer, SAM no conoce clases: solo separa
el objeto señalado.
""")
            gr.on(None, cajas_clasicas, E, [imagen_sam, caja], trigger_mode="always_last")
            prompt.change(lambda p: (gr.Dropdown(visible=p == PROMPTS_SAM[1]), gr.Button(visible=p == PROMPTS_SAM[1])),
                          prompt, [caja, boton_caja])
            imagen_sam.select(sam_punto, E + [nombre_sam, prompt], [gal_sam, tarjetas_sam, info_sam])
            boton_caja.click(sam_caja, E + [nombre_sam, caja], [gal_sam, tarjetas_sam, info_sam])
