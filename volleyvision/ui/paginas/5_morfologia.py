import numpy as np
import streamlit as st

from volleyvision.core import caracterizacion as ca
from volleyvision.core import filtros as fi
from volleyvision.core import morfologia as mo
from volleyvision.core import segmentacion as sg
from volleyvision.core import umbralizacion as um
from volleyvision.ui import componentes as ui

ui.encabezado("5 · Morfología matemática", "Limpiar las máscaras del balón y de los jugadores.")
img, _ = ui.selector_imagen("mor")

c1, c2, c3 = st.columns(3)
fuente = c1.selectbox("Máscara de entrada", ["Jugadores (distancia al fondo + Otsu)", "Balón (HSV amarillo + azul)",
                                             "Otsu de la imagen en grises"], key="mor_fuente")
ruido = c2.slider("Ruido sal y pimienta extra (%)", 0.0, 10.0, 1.0, 0.5, key="mor_ruido")
if fuente.startswith("Jugadores"):
    mascara = sg.mascara_primer_plano_kmeans(img)[0]
    mascara[sg.mascara_lineas_blancas(img) > 0] = 0
elif fuente.startswith("Balón"):
    mascara = um.mascara_por_nombre(img, "Balón (amarillo)") | um.mascara_por_nombre(img, "Balón (azul)")
else:
    mascara = um.otsu_opencv(img)[1]
if ruido > 0:
    mascara = fi.agregar_ruido(mascara, "sal_pimienta", ruido / 100, semilla=1)

c1, c2, c3, c4 = st.columns(4)
operacion = c1.selectbox("Operación", mo.OPERACIONES, index=2, key="mor_op")
forma = c2.selectbox("Forma del elemento estructurante", list(mo.FORMAS), index=1, key="mor_forma")
tam = c3.slider("Tamaño", 1, 31, 5, 2, key="mor_tam")
iters = c4.slider("Iteraciones", 1, 5, 1, key="mor_iter")
ee = mo.elemento_estructurante(forma, tam)
resultado = mo.aplicar(mascara, operacion, ee, iters)

c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    ui.mostrar(mascara, f"Entrada: {mo.contar_componentes(mascara)} componentes")
with c2:
    ui.mostrar(resultado, f"{operacion}: {mo.contar_componentes(resultado)} componentes")
with c3:
    ui.mostrar(np.kron(ee * 255, np.ones((12, 12), np.uint8)), f"Elemento estructurante {tam}×{tam}")
    st.code("\n".join(" ".join(str(v) for v in fila) for fila in ee) if tam <= 9 else f"{tam}×{tam}")

st.markdown("#### Todas las operaciones con el mismo elemento estructurante")
ui.cuadricula({op: mo.aplicar(mascara, op, ee, 1) for op in mo.OPERACIONES[:6]}, 3)

st.markdown("#### Implementación manual vs OpenCV")
recorte = mascara[: min(240, mascara.shape[0]), : min(320, mascara.shape[1])]
iguales_e = np.array_equal(mo.erosion_manual(recorte, ee), mo.erosion(recorte, ee))
iguales_d = np.array_equal(mo.dilatacion_manual(recorte, ee), mo.dilatacion(recorte, ee))
st.write(f"Erosión manual = cv2.erode: **{'sí' if iguales_e else 'no'}** · "
         f"Dilatación manual = cv2.dilate: **{'sí' if iguales_d else 'no'}**")

st.markdown("#### Limpieza completa (apertura → cierre → relleno → filtro de área)")
a, b, c = st.columns(3)
ap = a.slider("Apertura", 1, 15, 3, 2, key="mor_ap")
ci = b.slider("Cierre", 1, 21, 7, 2, key="mor_ci")
area = c.slider("Área mínima (px)", 0, 500, 40, 10, key="mor_area")
limpia = mo.limpiar_mascara(mascara, ap, ci, forma, area, rellenar=fuente.startswith("Balón"))
cands = ca.detectar_balon_clasico(img) if fuente.startswith("Balón") else []
ui.lado_a_lado(mascara, limpia, "Antes", f"Después: {mo.contar_componentes(limpia)} componentes")
if cands:
    st.caption(f"Con la máscara limpia, el clasificador de la página 7 encuentra "
               f"{sum(c['es_balon'] for c in cands)} balón(es) entre {len(cands)} candidatos.")

ui.que_esta_pasando(r"""
Con un elemento estructurante $B$:

* **Erosión** $A\ominus B=\{z \mid B_z\subseteq A\}$: mínimo del vecindario; borra puntos sueltos y adelgaza.
* **Dilatación** $A\oplus B=\{z \mid \hat B_z\cap A\neq\emptyset\}$: máximo del vecindario; engrosa y rellena.
* **Apertura** $A\circ B=(A\ominus B)\oplus B$: quita objetos más pequeños que $B$ (el ruido) sin encoger
  los grandes. Es idempotente.
* **Cierre** $A\bullet B=(A\oplus B)\ominus B$: rellena huecos y une las franjas del balón.
* **Gradiente** $(A\oplus B)-(A\ominus B)$: el contorno. **Top-hat** $A - A\circ B$: detalles claros
  pequeños (las líneas).

Para los jugadores se usa además un **cierre con un rectángulo vertical**, que une camiseta,
pantaloneta y piernas porque una persona de pie es alargada. Código: `core/morfologia.py`.
""")
