import cv2
import numpy as np
import streamlit as st

from volleyvision.core import histogramas as hi
from volleyvision.core import transformers_models as tm
from volleyvision.core import segmentacion as sg
from volleyvision.ui import componentes as ui
from volleyvision.ui import modelos
from volleyvision.utils import visualizacion as vis

ui.encabezado("3 · Histogramas", "Distribución de color, corrección de iluminación y comparación de uniformes.")
img, nombre = ui.selector_imagen("his")

tab_hist, tab_ecu, tab_uni = st.tabs(["Histograma por canal", "Ecualización y CLAHE", "Uniformes de los equipos"])

with tab_hist:
    espacio = st.radio("Espacio de color", ["RGB", "HSV", "Gris"], horizontal=True, key="his_espacio")
    c1, c2 = st.columns(2)
    with c1:
        ui.mostrar(img, "Imagen")
    with c2:
        ui.figura(vis.grafica_histograma(hi.calcular_histograma(img, espacio), f"Histograma {espacio}"))
    if espacio == "HSV":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ui.cuadricula({"H (tono)": hsv[..., 0], "S (saturación)": hsv[..., 1], "V (brillo)": hsv[..., 2]}, 3)
    ui.que_esta_pasando(r"""
El histograma cuenta cuántos píxeles tienen cada nivel: $h(r_k) = n_k$, y normalizado
$p(r_k) = n_k / N$. En **HSV** el tono H (0–179 en OpenCV) separa el color del brillo V: el pico
naranja de la cancha y el verde de la zona libre aparecen en H aunque la iluminación cambie.

Código: `core/histogramas.py` → `calcular_histograma` (usa `cv2.calcHist`).
""")

with tab_ecu:
    c1, c2 = st.columns(2)
    clip = c1.slider("CLAHE: clip limit", 0.5, 10.0, 2.0, 0.5, key="his_clip")
    baldosas = c2.slider("CLAHE: baldosas por lado", 2, 16, 8, key="his_tiles")
    glob = hi.ecualizar_global(img)
    clahe = hi.aplicar_clahe(img, clip, baldosas)
    ui.cuadricula({"Original": img, "Ecualización global (V de HSV)": glob, "CLAHE (L de Lab)": clahe}, 3)
    cols = st.columns(3)
    for col, (titulo, im) in zip(cols, [("Original", img), ("Global", glob), ("CLAHE", clahe)]):
        with col:
            ui.figura(vis.grafica_histograma(hi.calcular_histograma(im, "Gris"), titulo))
            st.metric("Contraste RMS", f"{hi.contraste_rms(im):.1f}")
            st.metric("Desv. del brillo entre bloques", f"{hi.uniformidad_iluminacion(im):.1f}",
                      help="Menor = iluminación más pareja en la imagen")
    ui.que_esta_pasando(r"""
**Ecualización global**: se usa la CDF como transformación,
$s_k = (L-1)\sum_{j\le k} p(r_j)$, para repartir los niveles por todo el rango. En color se aplica solo
a la luminancia para no alterar los tonos.

**CLAHE**: divide la imagen en baldosas, ecualiza cada una con su histograma **recortado** en el *clip
limit* (el exceso se reparte, así no se amplifica el ruido) e interpola entre baldosas. Corrige la
iluminación desigual del coliseo: compara la desviación del brillo entre bloques.

Código: `core/histogramas.py` → `ecualizar_manual` (CDF a mano), `ecualizar_global`, `aplicar_clahe`.
""")

with tab_uni:
    c1, c2 = st.columns(2)
    fuente = c1.radio("¿Cómo encontrar a los jugadores?", ["Segmentación clásica", "Detector transformer"],
                      key="his_fuente", horizontal=True)
    metrica = c2.selectbox("Métrica de comparación", list(hi.METODOS_COMPARACION), key="his_metrica")
    if fuente == "Segmentación clásica":
        cajas = [r["caja"] for r in sg.segmentar_jugadores_clasico(img)["regiones"]]
    else:
        nombre_det = modelos.selector_detector("his_det")
        det = modelos.obtener_detector(nombre_det)
        cajas = [d["caja"] for d in tm.detectar_objetos(img, det, 0.6)["detecciones"] if d["etiqueta"] == "person"]
    if len(cajas) < 2:
        st.warning("Se necesitan al menos 2 jugadores para comparar uniformes.")
    else:
        hists = hi.histogramas_de_cajas(img, cajas)
        equipos, M = hi.asignar_equipos(hists, metrica)
        colores = {0: vis.ROJO, 1: vis.VERDE}
        anotada = img.copy()
        for i, (caja, e) in enumerate(zip(cajas, equipos)):
            anotada = vis.dibujar_cajas(anotada, [{"caja": caja, "etiqueta": f"{i + 1}:{'AB'[e]}", "color": colores[e]}])
            x1, y1, x2, y2 = hi.recorte_camiseta(caja)
            cv2.rectangle(anotada, (x1, y1), (x2, y2), vis.NEUTRO, 1)
        c1, c2 = st.columns([3, 2])
        with c1:
            ui.mostrar(anotada, "Equipo A (rojo) / equipo B (verde); recuadro blanco = zona del torso usada")
        with c2:
            fig, ax = vis.figura(4.5, 4)
            im = ax.imshow(M, cmap="magma")
            ax.set_title(f"Matriz de {metrica}")
            ax.set_xticks(range(len(M)), [str(i + 1) for i in range(len(M))], fontsize=7)
            ax.set_yticks(range(len(M)), [str(i + 1) for i in range(len(M))], fontsize=7)
            fig.colorbar(im, ax=ax)
            ui.figura(fig)
            st.write(f"Equipo A: **{equipos.count(0)}** · Equipo B: **{equipos.count(1)}**")
        verdad = ui.verdad_muestra(nombre)
        if verdad is not None and fuente == "Segmentación clásica":
            st.caption("En la escena sintética el equipo A viste de rojo y el B de azul.")
    ui.que_esta_pasando(r"""
Para cada jugador se toma el **torso** (20–50 % de la altura de su caja) y se calcula el histograma 2D
de **tono y saturación** (H-S), ignorando píxeles grises u oscuros. Así no importa el brillo.

`cv2.compareHist` mide el parecido: **Correlación** (1 = iguales), **Chi-cuadrado** (0 = iguales),
**Intersección** $\sum\min(H_1,H_2)$ y **Bhattacharyya**
$d=\sqrt{1-\sum\sqrt{H_1H_2}/\sqrt{\bar H_1\bar H_2N^2}}$ (0 = iguales).

Asignación a equipos: se toman como referencia los dos jugadores **más distintos** y se itera
k-medias (k = 2) con esa métrica. Código: `core/histogramas.py` → `asignar_equipos`.
""")
