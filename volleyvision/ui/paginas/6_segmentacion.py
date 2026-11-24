import numpy as np
import pandas as pd
import streamlit as st

from volleyvision.core import segmentacion as sg
from volleyvision.core import transformers_models as tm
from volleyvision.ui import componentes as ui
from volleyvision.ui import modelos
from volleyvision.utils import io
from volleyvision.utils import visualizacion as vis

ui.encabezado("6 · Segmentación", "Métodos clásicos (contornos, watershed, k-means) y transformer (SegFormer, SAM).")
img, nombre = ui.selector_imagen("seg")

tabs = st.tabs(["K-means", "Contornos", "Watershed", "Jugadores (clásico)", "SegFormer", "SAM / SlimSAM"])

with tabs[0]:
    c1, c2 = st.columns(2)
    k = c1.slider("k (número de colores)", 2, 12, 5, key="seg_k")
    espacio = c2.radio("Espacio", list(sg.ESPACIOS_KMEANS), horizontal=True, key="seg_esp")
    km = sg.kmeans_color(img, k, espacio)
    ui.lado_a_lado(img, km["cuantizada"], "Original", f"K-means (k = {k}, {espacio})")
    ui.figura(vis.grafica_barras([str(i) for i in range(k)], list(km["proporciones"]), "Fracción del área por grupo",
                                 [("#%02x%02x%02x" % tuple(int(v) for v in c[::-1])) for c in km["centros_bgr"]]))
    ui.que_esta_pasando(r"""
K-means minimiza $J=\sum_j\sum_{x_i\in C_j}\lVert x_i-\mu_j\rVert^2$ alternando: asignar cada píxel al
centro más cercano y recalcular los centros. En **Lab** la distancia euclidiana se parece a la diferencia
de color percibida. Los grupos grandes son el fondo (cancha, zona libre, gradas); los jugadores y el
balón quedan en grupos pequeños o mezclados. Código: `core/segmentacion.py` → `kmeans_color`.
""")

jug = sg.segmentar_jugadores_clasico(img)

with tabs[1]:
    area_min = st.slider("Área mínima del contorno (px)", 0, 2000, 150, 10, key="seg_area")
    cs = sg.encontrar_contornos(jug["mascara"], area_min)
    ui.lado_a_lado(jug["mascara"], sg.dibujar_contornos(img, cs, con_caja=True), "Máscara de entrada",
                   f"{len(cs)} contornos")
    ui.que_esta_pasando(r"""
`cv2.findContours` (algoritmo de Suzuki–Abe) recorre la frontera de cada componente conexa de una
máscara binaria. Con `RETR_EXTERNAL` se toman solo los contornos exteriores y con `CHAIN_APPROX_SIMPLE`
se guardan solo los vértices. Si dos jugadores se tocan, salen **un solo** contorno: para eso está watershed.
""")

with tabs[2]:
    c1, c2 = st.columns(2)
    demo = c1.radio("Máscara", ["Jugadores de la imagen", "Demo: dos balones que se tocan"], key="seg_ws_demo")
    frac = c2.slider("Umbral del frente seguro (fracción del máximo de la distancia)", 0.1, 0.95, 0.6, 0.05,
                     key="seg_frac")
    if demo.startswith("Demo"):
        import cv2

        m = np.zeros((240, 360), np.uint8)
        cv2.circle(m, (130, 120), 70, 255, -1)
        cv2.circle(m, (255, 120), 60, 255, -1)
        base = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    else:
        m, base = jug["mascara"], img
    ws = sg.watershed_distancia(base, m, frac)
    ui.cuadricula({"Máscara": m, "Transformada de distancia": ws["distancia"], "Frente seguro (marcadores)":
                   ws["frente_seguro"], "Zona desconocida": ws["desconocido"],
                   "Etiquetas": vis.colorear_etiquetas(np.clip(ws["etiquetas"] - 1, 0, None)),
                   f"Watershed: {ws['n_objetos']} objetos": ws["resultado"]}, 3)
    st.caption(f"Contornos encontrarían {len(sg.encontrar_contornos(m))} objeto(s); watershed encuentra {ws['n_objetos']}.")
    ui.que_esta_pasando(r"""
1. **Transformada de distancia** $D(p)=\min_{q\notin A}\lVert p-q\rVert$: es máxima en el centro de cada objeto.
2. **Marcadores**: píxeles con $D \ge \alpha\cdot\max D$ de su componente (frente seguro).
3. **Desconocido** = fondo seguro (dilatación) − frente seguro.
4. `cv2.watershed` "inunda" el relieve desde cada marcador; donde se encuentran dos cuencas traza la
   frontera (en naranja). Separa objetos que se tocan.
""")

with tabs[3]:
    c1, c2, c3 = st.columns(3)
    kk = c1.slider("k del modelo de fondo", 3, 8, 6, key="seg_kk")
    cierre = c2.slider("Cierre vertical (px)", 3, 21, 9, 2, key="seg_cierre")
    usar_ws = c3.checkbox("Separar con watershed", False, key="seg_ws")
    r = sg.segmentar_jugadores_clasico(img, kk, tam_cierre=cierre, usar_watershed=usar_ws)
    verdad = ui.verdad_muestra(nombre)
    cajas = [x["caja"] for x in r["regiones"]]
    if verdad is not None:
        ev = sg.emparejar_cajas(cajas, verdad, 0.3)
        dibujo = vis.dibujar_cajas(img, [{"caja": c, "color": vis.VERDE if ok else vis.ROJO}
                                         for c, ok in zip(cajas, ev["correctas"])])
        st.write(f"Verdaderos positivos **{ev['VP']}** · falsos positivos **{ev['FP']}** · "
                 f"no detectados **{ev['FN']}** (verde = acierto, rojo = error)")
    else:
        dibujo = vis.dibujar_cajas(img, [{"caja": c} for c in cajas])
    ui.cuadricula({f"Distancia al fondo (T Otsu = {r['kmeans']['T_otsu']})": r["kmeans"]["distancia"],
                   "Máscara limpia": r["mascara"], f"{len(cajas)} jugadores": dibujo}, 3)
    ui.que_esta_pasando(r"""
Cadena clásica completa (sin aprendizaje):
k-means encuentra los colores del **fondo** → distancia de cada píxel al color de fondo más cercano →
**Otsu** → quitar líneas blancas → apertura + **cierre vertical** → contornos → filtro de forma
(alto/ancho ≥ 1.3). Funciona en la escena sintética, pero en fotos reales falla con el público, las
sombras y los jugadores superpuestos. Es la motivación para los transformers.
""")

with tabs[4]:
    if st.toggle("Ejecutar SegFormer", key="seg_segformer"):
        seg = modelos.obtener_segformer()
        sem = tm.segmentar_semantica(img, seg)
        pers = tm.mascara_de_clase(sem["etiquetas"], sem["id2label"], "person")
        ui.cuadricula({"Mapa semántico (ADE20K)": vis.colorear_etiquetas(sem["etiquetas"] + 1),
                       "Clase 'person'": vis.superponer_mascara(img, pers, vis.NARANJA, 0.55),
                       "Máscara clásica": jug["mascara"]}, 3)
        c1, c2 = st.columns(2)
        c1.metric("Tiempo de inferencia", f"{sem['tiempo_ms']:.0f} ms")
        c1.metric("IoU máscara clásica vs SegFormer (personas)", f"{sg.iou_mascaras(jug['mascara'], pers):.2f}")
        c2.dataframe(pd.DataFrame([{"clase": n, "área": f"{f:.1%}"} for n, f in sem["clases_presentes"][:10]]),
                     hide_index=True)
        st.caption(f"Modelo: [{seg.id_hf}]({seg.enlace}) · {seg.n_parametros / 1e6:.1f} M parámetros · {seg.dispositivo}")
    else:
        st.info(f"Activa el interruptor para descargar/cargar **{tm.MODELO_SEGFORMER}** (~15 MB).")
    ui.que_esta_pasando(r"""
**SegFormer** asigna una de las 150 clases de ADE20K a **cada píxel**. Su codificador es un transformer
jerárquico (*Mix Transformer*) que produce mapas a 1/4, 1/8, 1/16 y 1/32 de la resolución con
auto-atención eficiente $\mathrm{softmax}(QK^\top/\sqrt{d})V$ (las claves se reducen espacialmente), y un
decodificador MLP fusiona las escalas. No necesita codificación posicional.
""")

with tabs[5]:
    c1, c2 = st.columns([1, 2])
    with c1:
        nombre_sam = st.selectbox("Modelo", list(tm.MODELOS_SAM), key="seg_sam_modelo")
        prompt = st.radio("Prompt", ["Punto (clic)", "Caja (de la detección clásica)"], key="seg_sam_prompt")
        st.caption("Haz clic sobre el balón o un jugador en la imagen.")
    punto, caja = None, None
    with c2:
        if prompt.startswith("Punto"):
            from streamlit_image_coordinates import streamlit_image_coordinates

            v = streamlit_image_coordinates(io.bgr_a_rgb(img), width="stretch", key="seg_sam_clic")
            if v is not None:
                punto = (v["x"] * img.shape[1] / v["width"], v["y"] * img.shape[0] / v["height"])
        else:
            opciones = [x["caja"] for x in jug["regiones"]]
            if opciones:
                i = st.selectbox("Caja", range(len(opciones)), format_func=lambda j: f"Jugador {j + 1}: {opciones[j]}",
                                 key="seg_sam_caja")
                caja = opciones[i]
            else:
                st.warning("La segmentación clásica no encontró cajas.")
    if punto is not None or caja is not None:
        sam = modelos.obtener_sam(nombre_sam)
        r = tm.segmentar_con_sam(img, sam, punto=punto, caja=caja)
        base = vis.dibujar_puntos(img, [punto], vis.NARANJA, numerar=False) if punto else \
            vis.dibujar_cajas(img, [{"caja": caja}])
        ui.cuadricula({f"Máscara {i + 1} (IoU est. {p:.2f})": vis.superponer_mascara(base, m, vis.VERDE, 0.5)
                       for i, (m, p) in enumerate(zip(r["mascaras"], r["puntajes"]))}, 3)
        st.metric("Tiempo de inferencia", f"{r['tiempo_ms']:.0f} ms")
        st.caption(f"Modelo: [{sam.id_hf}]({sam.enlace}) · {sam.n_parametros / 1e6:.1f} M parámetros · {sam.dispositivo}")
    ui.que_esta_pasando(r"""
**SAM** (*Segment Anything*) segmenta "lo que se le señale": un codificador ViT procesa la imagen, un
codificador de *prompts* convierte el punto o la caja en tokens, y un decodificador ligero propone
**3 máscaras** (parte, objeto, todo) con su IoU estimado. **SlimSAM** es SAM podado y destilado
(~10 % de los parámetros), viable en CPU. A diferencia de SegFormer, SAM no conoce clases: solo separa
el objeto señalado.
""")
