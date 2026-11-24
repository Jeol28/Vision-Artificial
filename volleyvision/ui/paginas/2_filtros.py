import numpy as np
import streamlit as st

from volleyvision.core import filtros as fi
from volleyvision.ui import componentes as ui
from volleyvision.utils import visualizacion as vis

ui.encabezado("2 · Filtros", "Suavizado, detección de bordes y convolución manual aplicados a las líneas de la cancha.")
img, _ = ui.selector_imagen("fil")

tab_suave, tab_bordes, tab_conv, tab_lineas = st.tabs(
    ["Suavizado", "Bordes", "Convolución manual", "Líneas de la cancha"])

with tab_suave:
    c1, c2, c3 = st.columns(3)
    metodo = c1.selectbox("Filtro", fi.SUAVIZADOS, index=3, key="fil_metodo")
    k = c2.slider("Tamaño del kernel k (bilateral: diámetro d)", 1, 31, 9, 2, key="fil_k")
    ruido = c3.selectbox("Agregar ruido para comparar", ["Ninguno", "Gaussiano", "Sal y pimienta"], key="fil_ruido")
    sigma, sc, se = 0.0, 75.0, 75.0
    if metodo == "Gaussiano":
        sigma = st.slider("σ (0 = automático)", 0.0, 10.0, 0.0, 0.5, key="fil_sigma")
    if metodo == "Bilateral":
        a, b = st.columns(2)
        sc = a.slider("σ color (rango)", 1, 200, 75, key="fil_sc")
        se = b.slider("σ espacio", 1, 200, 75, key="fil_se")
    entrada = img
    if ruido == "Gaussiano":
        entrada = fi.agregar_ruido(img, "gaussiano", 0.08)
    elif ruido == "Sal y pimienta":
        entrada = fi.agregar_ruido(img, "sal_pimienta", 0.05)
    salida = fi.suavizar(entrada, metodo, k, sigma, sc, se)
    ui.lado_a_lado(entrada, salida, "Entrada", f"{metodo} (k = {k})")
    ui.que_esta_pasando(r"""
Un filtro lineal es una **convolución** $g(x,y)=\sum_{s,t} w(s,t)\,f(x-s, y-t)$.

* **Promedio**: $w = 1/k^2$. Reduce ruido pero borra bordes.
* **Gaussiano**: $w(s,t)\propto e^{-(s^2+t^2)/2\sigma^2}$. Pondera más el centro.
* **Mediana** (no lineal): la mediana del vecindario; elimina el ruido *sal y pimienta* sin difuminar.
* **Bilateral**: $w = e^{-\lVert p-q\rVert^2/2\sigma_s^2}\,e^{-|f_p-f_q|^2/2\sigma_r^2}$. Promedia solo
  píxeles **parecidos**: alisa el piso y conserva las líneas blancas. Prueba con ruido sal y pimienta: la
  mediana gana; con ruido gaussiano, el bilateral conserva mejor los bordes.

Código: `core/filtros.py` → `suavizar`, `filtro_bilateral`.
""")

with tab_bordes:
    c1, c2, c3, c4 = st.columns(4)
    ks = c1.select_slider("Kernel Sobel / Laplaciano", [1, 3, 5, 7], 3, key="fil_ks")
    t1 = c2.slider("Canny T_bajo", 0, 255, 50, key="fil_t1")
    t2 = c3.slider("Canny T_alto", 0, 255, 150, key="fil_t2")
    suav = c4.slider("Suavizado previo (Canny)", 0, 11, 5, 2, key="fil_suav")
    sob = fi.sobel(img, ks)
    ui.cuadricula({
        "Sobel Gx": vis.a_uint8(np.abs(sob["gx"])), "Sobel Gy": vis.a_uint8(np.abs(sob["gy"])),
        "Magnitud del gradiente": sob["magnitud"], "Laplaciano |∇²f|": fi.laplaciano(img, ks),
        f"Canny ({t1}, {t2})": fi.canny(img, t1, t2, 3, suav), "Original": img}, 3)
    ui.que_esta_pasando(r"""
* **Sobel**: $G_x = \begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix} * f$, $G_y = G_x^\top * f$,
  magnitud $\sqrt{G_x^2+G_y^2}$ y dirección $\theta=\operatorname{atan2}(G_y, G_x)$. Gx resalta las
  líneas verticales de la cancha; Gy, las horizontales.
* **Laplaciano**: $\nabla^2 f = f_{xx} + f_{yy}$, segunda derivada (sensible al ruido, por eso se suaviza antes).
* **Canny**: suavizado gaussiano → gradiente → supresión de no máximos (bordes de 1 px) → histéresis: un
  píxel es borde si su gradiente supera $T_{alto}$, o $T_{bajo}$ y está conectado a un borde fuerte.

Código: `core/filtros.py` → `sobel`, `laplaciano`, `canny`.
""")

with tab_conv:
    c1, c2 = st.columns([1, 2])
    with c1:
        preset = st.selectbox("Kernel predefinido", list(fi.KERNELS_PREDEFINIDOS), key="fil_preset")
        texto = st.text_area("Kernel (edítalo: filas en líneas, valores separados por espacios)",
                             fi.kernel_a_texto(fi.KERNELS_PREDEFINIDOS[preset]), height=140, key=f"fil_txt_{preset}")
        modo = st.radio("Visualización", ["recortar", "absoluto"], horizontal=True, key="fil_modo",
                        help="'absoluto' para kernels de bordes con valores negativos")
        usar_gris = st.checkbox("Trabajar en escala de grises", True, key="fil_gris")
    try:
        kernel = fi.kernel_desde_texto(texto)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    entrada = ui.a_gris(img) if usar_gris else img
    res = fi.comparar_convoluciones(entrada, kernel)
    with c2:
        m1, m2, m3 = st.columns(3)
        m1.metric("Manual (NumPy)", f"{res['tiempo_manual_ms']:.1f} ms")
        m2.metric("cv2.filter2D", f"{res['tiempo_opencv_ms']:.1f} ms")
        m3.metric("Diferencia máxima", f"{res['diferencia_max']:.1e}")
        st.caption(f"Kernel {kernel.shape[0]}×{kernel.shape[1]}, suma = {kernel.sum():.3f}")
    ui.cuadricula({"Entrada": entrada, "Convolución manual": fi.a_visualizable(res["manual"], modo),
                   "cv2.filter2D": fi.a_visualizable(res["opencv"], modo)}, 3)
    ui.que_esta_pasando(r"""
La implementación manual (`convolucion_manual`):

1. Rellena el borde por reflexión (`np.pad(..., mode="reflect")`, igual a `BORDER_REFLECT_101`).
2. Con `sliding_window_view` obtiene el vecindario $k_h\times k_w$ de cada píxel sin copiar memoria.
3. Calcula $g(x,y)=\sum_{s,t} w(s,t)\,f(x-s,y-t)$ con `einsum`, **volteando** el kernel 180°.

`cv2.filter2D` calcula **correlación** (no voltea), por eso para compararlas se le pasa el kernel
volteado. Las dos coinciden hasta el error de redondeo (~1e-12). OpenCV es más rápido porque usa
código optimizado (SIMD y, para kernels grandes, la DFT).
""")

with tab_lineas:
    c1, c2, c3, c4 = st.columns(4)
    d = c1.slider("Diámetro bilateral", 1, 15, 9, 2, key="lin_d")
    b = c2.slider("Canny T_bajo", 0, 255, 50, key="lin_t1")
    a = c3.slider("Canny T_alto", 0, 255, 150, key="lin_t2")
    votos = c4.slider("Votos Hough", 10, 300, 80, key="lin_votos")
    c5, c6 = st.columns(2)
    long_min = c5.slider("Longitud mínima del segmento (px)", 10, 300, 60, key="lin_long")
    hueco = c6.slider("Hueco máximo (px)", 1, 50, 10, key="lin_hueco")
    res = fi.resaltar_lineas_cancha(img, d, b, a, votos, long_min, hueco)
    ui.cuadricula({"Bilateral": res["suavizada"], "Canny": res["bordes"],
                   f"Hough: {len(res['segmentos'])} segmentos": res["lineas"]}, 3)
    ui.que_esta_pasando(r"""
Cadena para resaltar las líneas de la cancha:

1. **Bilateral**: elimina la textura del piso sin difuminar las líneas blancas.
2. **Canny**: bordes finos.
3. **Hough probabilística**: cada borde vota por las rectas $\rho = x\cos\theta + y\sin\theta$ que pasan
   por él; los máximos del acumulador con suficientes votos y longitud son segmentos rectos.

Código: `core/filtros.py` → `resaltar_lineas_cancha`.
""")
