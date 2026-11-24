"""2 · Filtros."""

import gradio as gr
import numpy as np

from volleyvision.core import filtros as fi
from volleyvision.ui import componentes as ui
from volleyvision.utils import visualizacion as vis

RUIDOS = ("Ninguno", "Gaussiano", "Sal y pimienta")


def suavizado(*args):
    *vals, metodo, k, ruido, sigma, sc, se = args
    img, _ = ui.resolver_imagen(*vals)
    entrada = img
    if ruido == "Gaussiano":
        entrada = fi.agregar_ruido(img, "gaussiano", 0.08)
    elif ruido == "Sal y pimienta":
        entrada = fi.agregar_ruido(img, "sal_pimienta", 0.05)
    salida = fi.suavizar(entrada, metodo, int(k), float(sigma), float(sc), float(se))
    return ui.rgb(entrada), gr.Image(ui.rgb(salida), label=f"{metodo} (k = {int(k)})")


def bordes(*args):
    *vals, ks, t1, t2, suav = args
    img, _ = ui.resolver_imagen(*vals)
    sob = fi.sobel(img, int(ks))
    return ui.galeria({
        "Sobel Gx": vis.a_uint8(np.abs(sob["gx"])), "Sobel Gy": vis.a_uint8(np.abs(sob["gy"])),
        "Magnitud del gradiente": sob["magnitud"], "Laplaciano |∇²f|": fi.laplaciano(img, int(ks)),
        f"Canny ({int(t1)}, {int(t2)})": fi.canny(img, int(t1), int(t2), 3, int(suav)), "Original": img})


def convolucion(*args):
    *vals, texto, modo, usar_gris = args
    img, _ = ui.resolver_imagen(*vals)
    try:
        kernel = fi.kernel_desde_texto(texto)
    except ValueError as e:
        raise gr.Error(str(e)) from e
    entrada = ui.a_gris(img) if usar_gris else img
    res = fi.comparar_convoluciones(entrada, kernel)
    tarjetas = ui.metricas({"Manual (NumPy)": f"{res['tiempo_manual_ms']:.1f} ms",
                            "cv2.filter2D": f"{res['tiempo_opencv_ms']:.1f} ms",
                            "Diferencia máxima": f"{res['diferencia_max']:.1e}",
                            "Kernel": f"{kernel.shape[0]}×{kernel.shape[1]}, suma = {kernel.sum():.3f}"})
    return tarjetas, ui.galeria({"Entrada": entrada, "Convolución manual": fi.a_visualizable(res["manual"], modo),
                                 "cv2.filter2D": fi.a_visualizable(res["opencv"], modo)})


def lineas(*args):
    *vals, d, b, a, votos, long_min, hueco = args
    img, _ = ui.resolver_imagen(*vals)
    res = fi.resaltar_lineas_cancha(img, int(d), int(b), int(a), int(votos), int(long_min), int(hueco))
    return ui.galeria({"Bilateral": res["suavizada"], "Canny": res["bordes"],
                       f"Hough: {len(res['segmentos'])} segmentos": res["lineas"]})


def construir(estado: gr.State) -> None:
    ui.encabezado("2 · Filtros", "Suavizado, detección de bordes y convolución manual aplicados a las líneas "
                                 "de la cancha.")
    sel = ui.selector_imagen()
    E = sel.entradas
    with gr.Tabs():
        with gr.Tab("Suavizado"):
            with gr.Row():
                metodo = gr.Dropdown(list(fi.SUAVIZADOS), value=fi.SUAVIZADOS[3], label="Filtro")
                k = gr.Slider(1, 31, 9, step=2, label="Tamaño del kernel k (bilateral: diámetro d)")
                ruido = gr.Dropdown(list(RUIDOS), value=RUIDOS[0], label="Agregar ruido para comparar")
            with gr.Row():
                sigma = gr.Slider(0, 10, 0, step=0.5, label="σ (0 = automático)", visible=False)
                sc = gr.Slider(1, 200, 75, step=1, label="σ color (rango)")
                se = gr.Slider(1, 200, 75, step=1, label="σ espacio")
            with gr.Row():
                entrada = ui.imagen_salida("Entrada")
                salida = ui.imagen_salida("Resultado")
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
            metodo.change(lambda m: (gr.Slider(visible=m == "Gaussiano"), gr.Slider(visible=m == "Bilateral"),
                                     gr.Slider(visible=m == "Bilateral")), metodo, [sigma, sc, se])
            gr.on(None, suavizado, E + [metodo, k, ruido, sigma, sc, se], [entrada, salida],
                  trigger_mode="always_last")

        with gr.Tab("Bordes"):
            with gr.Row():
                ks = gr.Radio([1, 3, 5, 7], value=3, label="Kernel Sobel / Laplaciano")
                t1 = gr.Slider(0, 255, 50, step=1, label="Canny T_bajo")
                t2 = gr.Slider(0, 255, 150, step=1, label="Canny T_alto")
                suav = gr.Slider(0, 11, 5, step=2, label="Suavizado previo (Canny)")
            gal_bordes = ui.nueva_galeria(3)
            ui.que_esta_pasando(r"""
* **Sobel**: $G_x = \begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix} * f$, $G_y = G_x^\top * f$,
  magnitud $\sqrt{G_x^2+G_y^2}$ y dirección $\theta=\operatorname{atan2}(G_y, G_x)$. Gx resalta las
  líneas verticales de la cancha; Gy, las horizontales.
* **Laplaciano**: $\nabla^2 f = f_{xx} + f_{yy}$, segunda derivada (sensible al ruido, por eso se suaviza antes).
* **Canny**: suavizado gaussiano → gradiente → supresión de no máximos (bordes de 1 px) → histéresis: un
  píxel es borde si su gradiente supera $T_{alto}$, o $T_{bajo}$ y está conectado a un borde fuerte.

Código: `core/filtros.py` → `sobel`, `laplaciano`, `canny`.
""")
            gr.on(None, bordes, E + [ks, t1, t2, suav], gal_bordes, trigger_mode="always_last")

        with gr.Tab("Convolución manual"):
            primero = next(iter(fi.KERNELS_PREDEFINIDOS))
            with gr.Row():
                with gr.Column(scale=1):
                    preset = gr.Dropdown(list(fi.KERNELS_PREDEFINIDOS), value=primero, label="Kernel predefinido")
                    texto = gr.Textbox(fi.kernel_a_texto(fi.KERNELS_PREDEFINIDOS[primero]), lines=5,
                                       label="Kernel (edítalo: filas en líneas, valores separados por espacios)")
                    modo = gr.Radio(["recortar", "absoluto"], value="recortar", label="Visualización",
                                    info="'absoluto' para kernels de bordes con valores negativos")
                    usar_gris = gr.Checkbox(True, label="Trabajar en escala de grises")
                with gr.Column(scale=2):
                    tarjetas = gr.HTML()
                    gal_conv = ui.nueva_galeria(3)
            ui.que_esta_pasando(r"""
La implementación manual (`convolucion_manual`):

1. Rellena el borde por reflexión (`np.pad(..., mode="reflect")`, igual a `BORDER_REFLECT_101`).
2. Con `sliding_window_view` obtiene el vecindario $k_h\times k_w$ de cada píxel sin copiar memoria.
3. Calcula $g(x,y)=\sum_{s,t} w(s,t)\,f(x-s,y-t)$ con `einsum`, **volteando** el kernel 180°.

`cv2.filter2D` calcula **correlación** (no voltea), por eso para compararlas se le pasa el kernel
volteado. Las dos coinciden hasta el error de redondeo (~1e-12). OpenCV es más rápido porque usa
código optimizado (SIMD y, para kernels grandes, la DFT).
""")
            preset.change(lambda p: fi.kernel_a_texto(fi.KERNELS_PREDEFINIDOS[p]), preset, texto)
            gr.on(None, convolucion, E + [texto, modo, usar_gris], [tarjetas, gal_conv], trigger_mode="always_last")

        with gr.Tab("Líneas de la cancha"):
            with gr.Row():
                d = gr.Slider(1, 15, 9, step=2, label="Diámetro bilateral")
                b = gr.Slider(0, 255, 50, step=1, label="Canny T_bajo")
                a = gr.Slider(0, 255, 150, step=1, label="Canny T_alto")
                votos = gr.Slider(10, 300, 80, step=1, label="Votos Hough")
            with gr.Row():
                long_min = gr.Slider(10, 300, 60, step=1, label="Longitud mínima del segmento (px)")
                hueco = gr.Slider(1, 50, 10, step=1, label="Hueco máximo (px)")
            gal_lineas = ui.nueva_galeria(3)
            ui.que_esta_pasando(r"""
Cadena para resaltar las líneas de la cancha:

1. **Bilateral**: elimina la textura del piso sin difuminar las líneas blancas.
2. **Canny**: bordes finos.
3. **Hough probabilística**: cada borde vota por las rectas $\rho = x\cos\theta + y\sin\theta$ que pasan
   por él; los máximos del acumulador con suficientes votos y longitud son segmentos rectos.

Código: `core/filtros.py` → `resaltar_lineas_cancha`.
""")
            gr.on(None, lineas, E + [d, b, a, votos, long_min, hueco], gal_lineas, trigger_mode="always_last")
