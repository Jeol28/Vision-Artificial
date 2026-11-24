import pandas as pd
import streamlit as st

from volleyvision.core import transformers_models as tm
from volleyvision.ui import componentes as ui
from volleyvision.ui import modelos
from volleyvision.utils import visualizacion as vis

ui.encabezado("8 · Transformers", "Detección de personas y balón con RT-DETR / DETR y comparación con el método clásico.")
img, nombre = ui.selector_imagen("tr", por_defecto="voleibol_03.jpg")

tab_det, tab_comp = st.tabs(["Detección", "Comparación clásico vs. transformer"])

with tab_det:
    c1, c2 = st.columns(2)
    with c1:
        nombre_det = modelos.selector_detector("tr_det")
    umbral = c2.slider("Umbral de confianza", 0.05, 0.95, 0.5, 0.05, key="tr_umbral")
    det = modelos.obtener_detector(nombre_det)
    res = tm.detectar_objetos(img, det, umbral)
    resumen = tm.resumen_detecciones(res)
    c1, c2 = st.columns([3, 1])
    with c1:
        ui.mostrar(vis.dibujar_cajas(img, tm.cajas_para_dibujar(res["detecciones"])),
                   "Naranja = persona · amarillo = balón (sports ball)")
    with c2:
        st.metric("Personas", resumen["personas"])
        st.metric("Balones", resumen["balones"])
        st.metric("Inferencia (red)", f"{res['tiempo_ms']:.0f} ms")
        st.metric("Total (con pre/posproceso)", f"{res['tiempo_total_ms']:.0f} ms")
        st.caption(f"[{det.id_hf}]({det.enlace}) · {det.n_parametros / 1e6:.1f} M parámetros · {det.dispositivo}")
    if res["detecciones"]:
        st.dataframe(pd.DataFrame([{"clase": d["etiqueta"], "confianza": round(d["puntaje"], 3),
                                    "caja (x1, y1, x2, y2)": tuple(round(v) for v in d["caja"])}
                                   for d in res["detecciones"]]), hide_index=True, width="stretch")
    ui.que_esta_pasando(r"""
**DETR** convierte la detección en predicción de un **conjunto**: CNN (ResNet-50) → codificador
transformer con auto-atención $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ (cada región "mira" toda la
imagen) → decodificador con 100 *object queries* que devuelven directamente (clase, caja). No usa
anclas ni NMS: en el entrenamiento el emparejamiento húngaro asigna cada query a un objeto.

**RT-DETR** mantiene la idea pero con un codificador híbrido eficiente y selección de queries por IoU:
es más rápido y más preciso. Ambos están entrenados en COCO, que tiene `person` y `sports ball`.
Se usan **solo en inferencia** con los pesos públicos.
""")

with tab_comp:
    c1, c2 = st.columns(2)
    umbral_c = c1.slider("Umbral de confianza (transformers)", 0.05, 0.95, 0.5, 0.05, key="tr_umbral_c")
    elegidos = c2.multiselect("Detectores", list(tm.MODELOS_DETECCION), list(tm.MODELOS_DETECCION), key="tr_sel")
    if st.button("Comparar", type="primary", key="tr_comparar"):
        detectores = {n: modelos.obtener_detector(n) for n in elegidos}
        verdad = ui.verdad_muestra(nombre)
        with st.spinner("Ejecutando los métodos…"):
            st.session_state["tr_comparacion"] = (nombre, tm.comparar_metodos(img, detectores, umbral_c, verdad))
    guardado = st.session_state.get("tr_comparacion")
    if guardado is None or guardado[0] != nombre:
        st.info("Pulsa **Comparar** para ejecutar el método clásico y los transformers sobre la misma imagen.")
    else:
        comp = guardado[1]
        cols = st.columns(len(comp))
        filas = []
        for col, (metodo, c) in zip(cols, comp.items()):
            dets = c["resultado"]["detecciones"]
            ev = c["evaluacion"]
            if ev is not None:  # verde = acierto, rojo = falso positivo (solo personas)
                personas = [d for d in dets if d["etiqueta"] == "person"]
                otras = [d for d in dets if d["etiqueta"] != "person"]
                cajas = tm.cajas_para_dibujar(personas, ev["correctas"]) + tm.cajas_para_dibujar(otras)
            else:
                cajas = tm.cajas_para_dibujar(dets)
            with col:
                st.markdown(f"**{metodo}**")
                ui.mostrar(vis.dibujar_cajas(img, cajas))
                st.metric("Tiempo", f"{c['resumen']['tiempo_ms']:.0f} ms")
                st.write(f"Personas: **{c['resumen']['personas']}** · Balones: **{c['resumen']['balones']}**")
            fila = {"método": metodo, "tiempo (ms)": round(c["resumen"]["tiempo_ms"], 1),
                    "personas": c["resumen"]["personas"], "balones": c["resumen"]["balones"]}
            if ev is not None:
                fila.update({"VP": ev["VP"], "FP": ev["FP"], "FN": ev["FN"],
                             "precisión": round(ev["precision"], 2), "exhaustividad": round(ev["exhaustividad"], 2)})
            filas.append(fila)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
            if comp and next(iter(comp.values()))["evaluacion"] is not None:
                st.caption("Escena sintética con verdad conocida: verde = persona bien detectada, rojo = falso positivo.")
        with c2:
            ui.figura(vis.grafica_barras([f["método"].split(" (")[0] for f in filas], [f["tiempo (ms)"] for f in filas],
                                         "Tiempo de inferencia", [vis.HEX["neutro"], vis.HEX["naranja"], "#4C8DFF"][:len(filas)],
                                         "ms"))
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
