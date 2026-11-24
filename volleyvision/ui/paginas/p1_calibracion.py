"""1 · Modelos de cámara y calibración."""

import copy

import gradio as gr
import numpy as np

from volleyvision.core import calibracion as cal
from volleyvision.ui import componentes as ui
from volleyvision.utils import io, sinteticos
from volleyvision.utils import visualizacion as vis

FUENTES_TABLERO = ("Tablero sintético (ejemplo)", "Subir fotos")

TEORIA_INTRINSECA = r"""
**Modelo pinhole.** Un punto 3D se proyecta como $s\,[u, v, 1]^\top = K\,[R \mid t]\,[X, Y, Z, 1]^\top$, con

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

($f_x, f_y$ focales en píxeles, $(c_x, c_y)$ punto principal) y $[R \mid t]$ la pose de la cámara.

**Distorsión radial** (barril si $k_1 < 0$): $x_d = x\,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + \dots$

**Método de Zhang.** Cada foto del tablero (plano $Z=0$) da una homografía $H = K[r_1\ r_2\ t]$; con
≥ 3 vistas se despeja $K$ y luego se refina todo minimizando el **error de reproyección**
$E = \sqrt{\tfrac{1}{N}\sum \lVert p_i - \hat p_i \rVert^2}$. Un error < 0.5 px es una buena calibración.

Código: `core/calibracion.py` → `calibrar_camara`, `corregir_distorsion`.
"""

TEORIA_HOMOGRAFIA = r"""
**Cámara trasera.** La cámara está detrás de tu línea de fondo y ve las 4 esquinas de **tu media
cancha** (9 × 9 m): las dos de la línea central, bajo la red, y las dos de tu línea de fondo.

**Homografía.** El piso es un plano, así que píxeles y metros se relacionan con una matriz $3\times3$:

$$\lambda \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} = H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

$H$ tiene 8 grados de libertad: cada esquina aporta 2 ecuaciones, por eso bastan las **4 esquinas
de la media cancha**, que en metros son (0, 9), (9, 9), (9, 18) y (0, 18): $x$ va de izquierda a
derecha y $y$ desde la línea de fondo rival (0) hasta la tuya (18), con la red en $y = 9$. Se resuelve
con DLT (vector singular de menor valor de $A h = 0$) y `cv2.warpPerspective` remuestrea la imagen.
La homografía vale para **todo** el plano del piso, así que también ubica a los rivales al otro lado de
la red, aunque con menos precisión porque están más lejos (cada píxel abarca más metros).

Solo los puntos **sobre el piso** se proyectan bien: por eso de los jugadores se usa el punto de los
pies, no el centroide. Código: `core/calibracion.py` → `homografia_cancha`, `homografia_dlt`, `vista_cenital`.
"""


def _imagenes_tablero(fuente, archivos) -> list[np.ndarray]:
    if fuente == FUENTES_TABLERO[1]:
        rutas = [a if isinstance(a, str) else a.name for a in archivos or []]
        return [io.cargar_imagen(r) for r in rutas]
    return [io.cargar_imagen(p) for p in io.listar_tableros()]


def calibrar(fuente, archivos, cols, filas, mm, estado):
    """Calibra con las fotos del tablero y guarda el resultado en el estado de la sesión."""
    imagenes = _imagenes_tablero(fuente, archivos)
    if len(imagenes) < 3:
        raise gr.Error("Se necesitan al menos 3 fotos del tablero.")
    patron = (int(cols), int(filas))
    try:
        res = cal.calibrar_camara(imagenes, patron, float(mm))
    except ValueError as e:
        raise gr.Error(str(e)) from e
    estado = copy.deepcopy(estado or ui.estado_nuevo())
    estado.update(calibracion=res, calibracion_imagenes=imagenes, calibracion_patron=patron)

    texto_k = (r"**Matriz intrínseca K**" "\n\n"
               r"$$K = \begin{bmatrix} %.1f & %.1f & %.1f \\ 0 & %.1f & %.1f \\ 0 & 0 & 1 \end{bmatrix}$$"
               % (res.K[0, 0], res.K[0, 1], res.K[0, 2], res.K[1, 1], res.K[1, 2]))
    if fuente == FUENTES_TABLERO[0]:
        Kv = sinteticos.K_VERDADERA
        texto_k += (f"\n\n<span class='vv-nota'>Verdadera: fx = {Kv[0, 0]:.0f}, fy = {Kv[1, 1]:.0f}, "
                    f"cx = {Kv[0, 2]:.0f}, cy = {Kv[1, 2]:.0f}</span>")
    texto_k += ("\n\n**Coeficientes de distorsión** (k₁, k₂, p₁, p₂, k₃)\n\n```\n"
                + np.array2string(res.dist, precision=4, suppress_small=True) + "\n```")
    tarjetas = ui.metricas({"Error de reproyección (RMS)": f"{res.error_rms:.3f} px",
                            "Vistas válidas": f"{len(res.indices_validos)} / {len(imagenes)}",
                            "Tamaño de imagen": f"{res.tam_imagen[0]}×{res.tam_imagen[1]}"})
    fig = vis.grafica_barras([str(i) for i in res.indices_validos], res.errores_por_imagen,
                             "Error de reproyección por vista", ylabel="px")
    muestra = {f"Vista {i}": cal.dibujar_esquinas(imagenes[i], patron, cal.encontrar_esquinas(imagenes[i], patron)[1])
               for i in res.indices_validos[:3]}
    vistas = gr.Dropdown(choices=res.indices_validos, value=res.indices_validos[0], interactive=True)
    return tarjetas, texto_k, ui.figura(fig), ui.galeria(muestra), vistas, estado


def corregir(vista, alfa, estado):
    """Imagen con distorsión y corregida con la calibración guardada."""
    res = (estado or {}).get("calibracion")
    if res is None or vista is None:
        return None, None
    img = estado["calibracion_imagenes"][int(vista)]
    corregida, _ = cal.corregir_distorsion(img, res.K, res.dist, float(alfa))
    return ui.rgb(img), ui.rgb(corregida)


def homografia(*args):
    """Vista cenital de la cancha a partir de las esquinas marcadas."""
    *vals, ppm, margen, estado = args
    img, nombre = ui.resolver_imagen(*vals)
    esquinas = ui.esquinas_de(estado, nombre)
    if esquinas is None:
        return None, ("Marca las 4 esquinas de tu media cancha para ver la cancha desde arriba. "
                      "Quedan guardadas para las pestañas de Caracterización y Pipeline.")
    hom = cal.homografia_cancha(esquinas, int(ppm), float(margen))
    H_dlt = cal.homografia_dlt(esquinas, cal.ESQUINAS_MEDIA_CANCHA_M)
    dif = np.abs(H_dlt - hom.H_metros / hom.H_metros[2, 2]).max()
    texto = ("**Homografía imagen → metros**\n\n```\n"
             + np.array2string(hom.H_metros, precision=5, suppress_small=True)
             + f"\n```\n\n<span class='vv-nota'>DLT propio vs OpenCV: diferencia máx. = {dif:.2e}</span>")
    return ui.rgb(cal.vista_cenital(img, hom)), texto


def construir(estado: gr.State) -> None:
    ui.encabezado("1 · Modelos de cámara y calibración",
                  "Calibración intrínseca con tablero de ajedrez y homografía de la cancha desde la cámara trasera.")
    ui.en_pocas_palabras("toda cámara deforma un poco la imagen. Con fotos de un tablero de ajedrez se mide esa "
                         "deformación para corregirla, y con las 4 esquinas de la cancha se calcula cómo se vería "
                         "desde arriba.")
    with gr.Tabs():
        with gr.Tab("Calibración intrínseca"):
            fuente = gr.Radio(list(FUENTES_TABLERO), value=FUENTES_TABLERO[0], label="Fotos del tablero")
            with ui.opciones_avanzadas():
                with gr.Row():
                    cols = gr.Number(sinteticos.PATRON[0], label="Esquinas interiores (columnas)", precision=0,
                                     minimum=3, maximum=20)
                    filas = gr.Number(sinteticos.PATRON[1], label="Esquinas interiores (filas)", precision=0,
                                      minimum=3, maximum=20)
                    mm = gr.Number(sinteticos.TAM_CUADRO_MM, label="Lado del cuadro (mm)", minimum=1, maximum=200)
            archivos = gr.File(label="Fotos del tablero (mínimo 3, ideal 10–20 desde ángulos distintos)",
                               file_count="multiple", file_types=["image"], visible=False)
            ui.md(f"<span class='vv-nota'>{len(io.listar_tableros())} vistas sintéticas generadas con K y "
                  f"distorsión conocidas (k₁ = {sinteticos.DIST_VERDADERA[0]}, "
                  f"k₂ = {sinteticos.DIST_VERDADERA[1]}).</span>")
            boton = gr.Button("Calibrar cámara", variant="primary")
            tarjetas = gr.HTML(ui.metricas({"Estado": "Pulsa «Calibrar cámara»"}))
            with gr.Row():
                texto_k = ui.md()
                grafica = gr.Plot(show_label=False)
            ui.md("#### Esquinas detectadas")
            esquinas = ui.nueva_galeria(3)
            ui.md("#### Corrección de distorsión")
            with gr.Row():
                vista = gr.Dropdown([], label="Vista")
                alfa = gr.Slider(0, 1, 0, step=0.05, label="α (0 = recortar, 1 = conservar todo)")
            with gr.Row():
                antes = ui.imagen_salida("Con distorsión (rectas curvadas)")
                despues = ui.imagen_salida("Corregida (cv2.undistort)")
            ui.que_esta_pasando(TEORIA_INTRINSECA)

            fuente.change(lambda f: gr.File(visible=f == FUENTES_TABLERO[1]), fuente, archivos)
            boton.click(calibrar, [fuente, archivos, cols, filas, mm, estado],
                        [tarjetas, texto_k, grafica, esquinas, vista, estado])
            gr.on([vista.change, alfa.change], corregir, [vista, alfa, estado], [antes, despues],
                  trigger_mode="always_last")

        with gr.Tab("Homografía de la cancha"):
            sel = ui.selector_imagen()
            with gr.Row():
                with gr.Column(scale=3):
                    ui.selector_esquinas(sel, estado)
                with gr.Column(scale=2):
                    with ui.opciones_avanzadas():
                        ppm = gr.Slider(10, 60, 30, step=1, label="Resolución de la vista cenital (px/m)")
                        margen = gr.Slider(0, 5, 2, step=0.5, label="Zona libre alrededor (m)")
                    cenital = ui.imagen_salida("Vista cenital (cv2.warpPerspective)")
                    texto_h = ui.md()
            ui.que_esta_pasando(TEORIA_HOMOGRAFIA)
            gr.on(None, homografia, sel.entradas + [ppm, margen, estado], [cenital, texto_h],
                  trigger_mode="always_last")
