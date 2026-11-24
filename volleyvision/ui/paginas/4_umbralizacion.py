import numpy as np
import streamlit as st

from volleyvision.core import umbralizacion as um
from volleyvision.ui import componentes as ui
from volleyvision.utils import visualizacion as vis

ui.encabezado("4 · Umbralización y binarización", "Separar el balón, la cancha y los jugadores del fondo.")
img, _ = ui.selector_imagen("umb")
gris = ui.a_gris(img)

tab_glob, tab_otsu, tab_adap, tab_hsv = st.tabs(["Global", "Otsu", "Adaptativo", "Rango de color HSV"])

with tab_glob:
    c1, c2 = st.columns([3, 1])
    T = c1.slider("Umbral T", 0, 255, 127, key="umb_T")
    inv = c2.checkbox("Invertir", key="umb_inv")
    ui.lado_a_lado(gris, um.umbral_global(gris, T, inv), "Grises", f"Binaria (T = {T})")
    ui.que_esta_pasando(r"""
$$g(x,y)=\begin{cases}255 & f(x,y) > T\\ 0 & f(x,y)\le T\end{cases}$$

Un solo umbral para toda la imagen: funciona si el objeto y el fondo tienen niveles bien separados y la
iluminación es uniforme. Código: `core/umbralizacion.py` → `umbral_global`.
""")

with tab_otsu:
    T_m, curva = um.otsu_manual(gris)
    T_o, bin_o = um.otsu_opencv(gris)
    c1, c2, c3 = st.columns(3)
    c1.metric("T (Otsu manual)", T_m)
    c2.metric("T (cv2.THRESH_OTSU)", T_o)
    c3.metric("Diferencia", abs(T_m - T_o))
    c1, c2 = st.columns(2)
    with c1:
        ui.figura(vis.grafica_curva(np.arange(256), curva, "Varianza entre clases σ²_B(t)", "t", "σ²_B",
                                    {f"T manual = {T_m}": T_m, f"T OpenCV = {T_o}": T_o}))
    with c2:
        h = np.bincount(gris.ravel(), minlength=256)
        ui.figura(vis.grafica_curva(np.arange(256), h, "Histograma de grises", "nivel", "píxeles",
                                    {f"T* = {T_m}": T_m}))
    ui.lado_a_lado(um.umbral_global(gris, T_m), bin_o, "Otsu manual", "Otsu OpenCV")
    ui.que_esta_pasando(r"""
Otsu prueba todos los $t$ y elige el que **maximiza la varianza entre clases**:

$$\omega_0(t)=\sum_{i\le t}p_i,\quad \mu(t)=\sum_{i\le t} i\,p_i,\quad
\sigma_B^2(t)=\frac{\left(\mu_T\,\omega_0(t)-\mu(t)\right)^2}{\omega_0(t)\,(1-\omega_0(t))}$$

Como $\sigma_T^2=\sigma_W^2+\sigma_B^2$ es constante, maximizar $\sigma_B^2$ equivale a minimizar la
varianza **dentro** de cada clase. La implementación manual (`otsu_manual`) usa sumas acumuladas, así
que es $O(L)$ después del histograma, y coincide con OpenCV.
""")

with tab_adap:
    c1, c2, c3, c4 = st.columns(4)
    metodo = c1.radio("Método", ["gaussiano", "media"], key="umb_met")
    bloque = c2.slider("Tamaño de bloque", 3, 101, 31, 2, key="umb_bloque")
    C = c3.slider("Constante C", -20, 30, 5, key="umb_C")
    inv = c4.checkbox("Invertir", True, key="umb_inv2")
    ui.lado_a_lado(um.otsu_opencv(gris, inv)[1], um.umbral_adaptativo(gris, metodo, bloque, C, inv),
                   "Otsu global", f"Adaptativo ({metodo}, bloque {bloque}, C = {C})")
    ui.que_esta_pasando(r"""
Cada píxel tiene su propio umbral calculado en su vecindario $b\times b$:

$$T(x,y)=\operatorname{media}_{b\times b}(x,y) - C \quad\text{o}\quad
T(x,y)=\left(G_\sigma * f\right)(x,y) - C$$

Con la iluminación desigual de un coliseo (un lado más oscuro), el umbral global falla y el adaptativo
resalta las líneas y los bordes de los jugadores en toda la imagen.
""")

with tab_hsv:
    c1, c2 = st.columns([1, 3])
    with c1:
        preset = st.selectbox("Rango predefinido", list(um.RANGOS_HSV), key="umb_preset")
        (h0, s0, v0), (h1, s1, v1) = um.RANGOS_HSV[preset]
        hr = st.slider("H (tono)", 0, 179, (h0, h1), key=f"umb_h_{preset}")
        sr = st.slider("S (saturación)", 0, 255, (s0, s1), key=f"umb_s_{preset}")
        vr = st.slider("V (brillo)", 0, 255, (v0, v1), key=f"umb_v_{preset}")
        circular = st.checkbox("Tono circular (H_min > H_max, p. ej. rojos)", key="umb_circ")
    bajo = (hr[1] if circular else hr[0], sr[0], vr[0])
    alto = (hr[0] if circular else hr[1], sr[1], vr[1])
    m = um.umbral_hsv(img, bajo, alto)
    with c2:
        ui.lado_a_lado(m, vis.superponer_mascara(img, m, vis.NARANJA, 0.6), f"Máscara ({um.fraccion_blanca(m):.1%})",
                       "Superpuesta")
    ui.que_esta_pasando(r"""
En HSV un color es un intervalo $[H_{min},H_{max}]\times[S_{min},S_{max}]\times[V_{min},V_{max}]$
(`cv2.inRange`). El tono H no depende del brillo, así que el amarillo del balón o el naranja de la cancha
se aíslan aunque haya sombras. En OpenCV $H\in[0,179]$ (grados/2); el rojo está en ambos extremos,
por eso existe el rango **circular**. La máscara queda con ruido y huecos: se limpia con **morfología**
(siguiente página).
""")
