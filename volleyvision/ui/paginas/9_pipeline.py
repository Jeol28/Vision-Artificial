import pandas as pd
import streamlit as st

from volleyvision.core import pipeline as pl
from volleyvision.core import transformers_models as tm
from volleyvision.ui import componentes as ui
from volleyvision.ui import modelos
from volleyvision.utils import visualizacion as vis

ui.encabezado("9 · Pipeline completo", "Todas las etapas encadenadas sobre una imagen o un frame de video.")
img, nombre = ui.selector_imagen("pip")

with st.expander("Configuración", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Calibración y cancha**")
        calib = st.session_state.get("calibracion")
        usar_calib = st.checkbox("Corregir distorsión con la calibración", calib is not None,
                                 disabled=calib is None, key="pip_calib",
                                 help="Calibra primero en la página 1." if calib is None else None)
        esquinas = ui.esquinas_de(nombre)
        if esquinas is None and ui.esquinas_ejemplo(nombre, img) is not None:
            if st.checkbox("Usar esquinas de ejemplo de la escena sintética", True, key="pip_ej"):
                esquinas = ui.esquinas_ejemplo(nombre, img)
        st.write("Esquinas de la media cancha: " + ("✅ listas" if esquinas is not None else "❌ sin marcar"))
        marcar = st.checkbox("Marcar esquinas aquí", esquinas is None, key="pip_marcar")
    with c2:
        st.markdown("**Clásico**")
        clip = st.slider("CLAHE clip limit", 0.5, 6.0, 2.0, 0.5, key="pip_clip")
        k = st.slider("k del modelo de fondo", 3, 8, 6, key="pip_k")
        cierre = st.slider("Cierre vertical (px)", 3, 21, 9, 2, key="pip_cierre")
    with c3:
        st.markdown("**Transformers**")
        usar_det = st.checkbox("Detector transformer", True, key="pip_det")
        nombre_det = modelos.selector_detector("pip_det_sel") if usar_det else None
        umbral = st.slider("Umbral de confianza", 0.1, 0.95, 0.5, 0.05, key="pip_umbral")
        usar_seg = st.checkbox("SegFormer", False, key="pip_seg")

if marcar:
    esquinas = ui.selector_esquinas(img, nombre, "pip_esq") if esquinas is None else esquinas
    if ui.esquinas_de(nombre) is None and esquinas is None:
        st.caption("Marca las 4 esquinas o desmarca la opción para ejecutar sin mapa cenital.")

if st.button("▶ Ejecutar pipeline", type="primary"):
    detector = modelos.obtener_detector(nombre_det) if usar_det else None
    segformer = modelos.obtener_segformer() if usar_seg else None
    cfg = pl.ConfigPipeline(calibracion=calib if usar_calib else None, esquinas=esquinas, clip_limit=clip,
                            k_kmeans=k, tam_cierre=cierre, umbral_deteccion=umbral)
    with st.spinner("Procesando las 7 etapas…"):
        st.session_state["pip_resultado"] = (nombre, pl.ejecutar_pipeline(img, cfg, detector, segformer))

guardado = st.session_state.get("pip_resultado")
if guardado is None or guardado[0] != nombre:
    st.info("Configura y pulsa **Ejecutar pipeline**.")
    st.stop()

pasos = guardado[1]
total = sum(p.tiempo_ms for p in pasos)
c1, c2 = st.columns([2, 3])
c1.metric("Tiempo total", f"{total:.0f} ms")
with c2:
    ui.figura(vis.grafica_barras([p.nombre.split(".")[0] for p in pasos], [p.tiempo_ms for p in pasos],
                                 "Tiempo por etapa", ylabel="ms"))

for paso in pasos:
    st.markdown(f"### {paso.nombre} <span style='color:#B8BFCC;font-size:0.9rem'>· {paso.tiempo_ms:.0f} ms</span>",
                unsafe_allow_html=True)
    if paso.nota:
        st.caption(paso.nota)
    if paso.imagenes:
        ui.cuadricula(paso.imagenes, min(3, len(paso.imagenes)) if len(paso.imagenes) != 4 else 2)
    tablas = {k: v for k, v in paso.datos.items() if isinstance(v, list)}
    escalares = {k: v for k, v in paso.datos.items() if not isinstance(v, list)}
    if escalares:
        cols = st.columns(min(4, len(escalares)))
        for i, (k, v) in enumerate(escalares.items()):
            texto = f"{v:.2f}" if isinstance(v, float) else (", ".join(f"{a}: {b}" for a, b in v.items())
                                                             if isinstance(v, dict) else str(v))
            cols[i % len(cols)].metric(k, texto)
    for k, v in tablas.items():
        if v:
            with st.expander(k.replace("_", " ").capitalize()):
                st.dataframe(pd.DataFrame(v), hide_index=True, width="stretch")
    st.divider()

ui.que_esta_pasando(f"""
El pipeline (`core/pipeline.py` → `ejecutar_pipeline`) encadena los temas del curso:

1. **Calibración**: `cv2.undistort` con la K y la distorsión estimadas (las esquinas marcadas también se corrigen).
2. **Histograma**: CLAHE sobre L de Lab para emparejar la iluminación.
3. **Filtros**: bilateral + Canny + Hough para las líneas de la cancha.
4. **Umbralización + morfología**: distancia al fondo con Otsu, rangos HSV del balón, apertura y cierre.
5. **Segmentación**: contornos de la máscara clásica y, si se activa, {', '.join(tm.MODELOS_DETECCION)} / SegFormer.
6. **Caracterización**: descriptores, reglas "¿es balón?" y equipos por histogramas H-S.
7. **Mapa cenital**: homografía (a partir de tu media cancha) aplicada al punto de apoyo de cada jugador;
   tu campo queda abajo y el rival arriba, como lo ve la cámara trasera.

Si hay detector transformer y encuentra personas, el mapa usa sus cajas; si no, las del método clásico.
""")
