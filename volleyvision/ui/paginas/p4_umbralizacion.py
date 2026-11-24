"""4 · Umbralización y binarización."""

import gradio as gr
import numpy as np

from volleyvision.core import umbralizacion as um
from volleyvision.ui import componentes as ui
from volleyvision.utils import visualizacion as vis


def _gris(vals):
    img, _ = ui.resolver_imagen(*vals)
    return img, ui.a_gris(img)


def umbral_global(*args):
    *vals, T, inv = args
    _, gris = _gris(vals)
    return ui.rgb(gris), gr.Image(ui.rgb(um.umbral_global(gris, int(T), bool(inv))),
                                  label=f"Binaria (T = {int(T)})")


def otsu(*args):
    _, gris = _gris(args)
    T_m, curva = um.otsu_manual(gris)
    T_o, bin_o = um.otsu_opencv(gris)
    tarjetas = ui.metricas({"T (Otsu manual)": T_m, "T (cv2.THRESH_OTSU)": T_o, "Diferencia": abs(T_m - T_o)})
    f1 = vis.grafica_curva(np.arange(256), curva, "Varianza entre clases σ²_B(t)", "t", "σ²_B",
                           {f"T manual = {T_m}": T_m, f"T OpenCV = {T_o}": T_o})
    h = np.bincount(gris.ravel(), minlength=256)
    f2 = vis.grafica_curva(np.arange(256), h, "Histograma de grises", "nivel", "píxeles", {f"T* = {T_m}": T_m})
    return tarjetas, ui.figura(f1), ui.figura(f2), ui.rgb(um.umbral_global(gris, T_m)), ui.rgb(bin_o)


def adaptativo(*args):
    *vals, metodo, bloque, C, inv = args
    _, gris = _gris(vals)
    return (ui.rgb(um.otsu_opencv(gris, bool(inv))[1]),
            gr.Image(ui.rgb(um.umbral_adaptativo(gris, metodo, int(bloque), float(C), bool(inv))),
                     label=f"Adaptativo ({metodo}, bloque {int(bloque)}, C = {C})"))


def rango_preset(preset):
    """Valores de los sliders HSV para un rango predefinido."""
    (h0, s0, v0), (h1, s1, v1) = um.RANGOS_HSV[preset]
    return h0, h1, s0, s1, v0, v1


def hsv(*args):
    *vals, hmin, hmax, smin, smax, vmin, vmax, circular = args
    img, _ = ui.resolver_imagen(*vals)
    hmin, hmax = int(hmin), int(hmax)
    bajo = (hmax if circular else hmin, int(smin), int(vmin))
    alto = (hmin if circular else hmax, int(smax), int(vmax))
    m = um.umbral_hsv(img, bajo, alto)
    return (gr.Image(ui.rgb(m), label=f"Máscara ({um.fraccion_blanca(m):.1%})"),
            ui.rgb(vis.superponer_mascara(img, m, vis.NARANJA, 0.6)))


def construir(estado: gr.State) -> None:
    ui.encabezado("4 · Umbralización y binarización", "Separar el balón, la cancha y los jugadores del fondo.")
    sel = ui.selector_imagen()
    E = sel.entradas
    with gr.Tabs():
        with gr.Tab("Global"):
            with gr.Row():
                T = gr.Slider(0, 255, 127, step=1, label="Umbral T", scale=3)
                inv = gr.Checkbox(False, label="Invertir", scale=1)
            with gr.Row():
                gris = ui.imagen_salida("Grises")
                binaria = ui.imagen_salida("Binaria")
            ui.que_esta_pasando(r"""
$$g(x,y)=\begin{cases}255 & f(x,y) > T\\ 0 & f(x,y)\le T\end{cases}$$

Un solo umbral para toda la imagen: funciona si el objeto y el fondo tienen niveles bien separados y la
iluminación es uniforme. Código: `core/umbralizacion.py` → `umbral_global`.
""")
            gr.on(None, umbral_global, E + [T, inv], [gris, binaria], trigger_mode="always_last")

        with gr.Tab("Otsu"):
            tarjetas = gr.HTML()
            with gr.Row():
                curva = gr.Plot(show_label=False)
                hist = gr.Plot(show_label=False)
            with gr.Row():
                manual = ui.imagen_salida("Otsu manual")
                opencv = ui.imagen_salida("Otsu OpenCV")
            ui.que_esta_pasando(r"""
Otsu prueba todos los $t$ y elige el que **maximiza la varianza entre clases**:

$$\omega_0(t)=\sum_{i\le t}p_i,\quad \mu(t)=\sum_{i\le t} i\,p_i,\quad
\sigma_B^2(t)=\frac{\left(\mu_T\,\omega_0(t)-\mu(t)\right)^2}{\omega_0(t)\,(1-\omega_0(t))}$$

Como $\sigma_T^2=\sigma_W^2+\sigma_B^2$ es constante, maximizar $\sigma_B^2$ equivale a minimizar la
varianza **dentro** de cada clase. La implementación manual (`otsu_manual`) usa sumas acumuladas, así
que es $O(L)$ después del histograma, y coincide con OpenCV.
""")
            gr.on(None, otsu, E, [tarjetas, curva, hist, manual, opencv], trigger_mode="always_last")

        with gr.Tab("Adaptativo"):
            with gr.Row():
                metodo = gr.Radio(["gaussiano", "media"], value="gaussiano", label="Método")
                bloque = gr.Slider(3, 101, 31, step=2, label="Tamaño de bloque")
                C = gr.Slider(-20, 30, 5, step=1, label="Constante C")
                inv2 = gr.Checkbox(True, label="Invertir")
            with gr.Row():
                glob = ui.imagen_salida("Otsu global")
                adap = ui.imagen_salida("Adaptativo")
            ui.que_esta_pasando(r"""
Cada píxel tiene su propio umbral calculado en su vecindario $b\times b$:

$$T(x,y)=\operatorname{media}_{b\times b}(x,y) - C \quad\text{o}\quad
T(x,y)=\left(G_\sigma * f\right)(x,y) - C$$

Con la iluminación desigual de un coliseo (un lado más oscuro), el umbral global falla y el adaptativo
resalta las líneas y los bordes de los jugadores en toda la imagen.
""")
            gr.on(None, adaptativo, E + [metodo, bloque, C, inv2], [glob, adap], trigger_mode="always_last")

        with gr.Tab("Rango de color HSV"):
            primero = next(iter(um.RANGOS_HSV))
            h0, h1, s0, s1, v0, v1 = rango_preset(primero)
            with gr.Row():
                with gr.Column(scale=1):
                    preset = gr.Dropdown(list(um.RANGOS_HSV), value=primero, label="Rango predefinido")
                    with gr.Row():
                        hmin = gr.Slider(0, 179, h0, step=1, label="H mín (tono)")
                        hmax = gr.Slider(0, 179, h1, step=1, label="H máx")
                    with gr.Row():
                        smin = gr.Slider(0, 255, s0, step=1, label="S mín (saturación)")
                        smax = gr.Slider(0, 255, s1, step=1, label="S máx")
                    with gr.Row():
                        vmin = gr.Slider(0, 255, v0, step=1, label="V mín (brillo)")
                        vmax = gr.Slider(0, 255, v1, step=1, label="V máx")
                    circular = gr.Checkbox(False, label="Tono circular (H mín > H máx, p. ej. rojos)")
                with gr.Column(scale=3):
                    with gr.Row():
                        mascara = ui.imagen_salida("Máscara")
                        superpuesta = ui.imagen_salida("Superpuesta")
            ui.que_esta_pasando(r"""
En HSV un color es un intervalo $[H_{min},H_{max}]\times[S_{min},S_{max}]\times[V_{min},V_{max}]$
(`cv2.inRange`). El tono H no depende del brillo, así que el amarillo del balón o el naranja de la cancha
se aíslan aunque haya sombras. En OpenCV $H\in[0,179]$ (grados/2); el rojo está en ambos extremos,
por eso existe el rango **circular**. La máscara queda con ruido y huecos: se limpia con **morfología**
(siguiente pestaña).
""")
            preset.change(rango_preset, preset, [hmin, hmax, smin, smax, vmin, vmax])
            gr.on(None, hsv, E + [hmin, hmax, smin, smax, vmin, vmax, circular], [mascara, superpuesta],
                  trigger_mode="always_last")
