"""3 · Histogramas."""

import cv2
import gradio as gr

from volleyvision.core import histogramas as hi
from volleyvision.core import segmentacion as sg
from volleyvision.core import transformers_models as tm
from volleyvision.ui import componentes as ui
from volleyvision.ui import modelos
from volleyvision.utils import visualizacion as vis

FUENTES_JUGADORES = ("Segmentación clásica", "Detector transformer")


def histograma(*args):
    *vals, espacio = args
    img, _ = ui.resolver_imagen(*vals)
    fig = ui.figura(vis.grafica_histograma(hi.calcular_histograma(img, espacio), f"Histograma {espacio}"))
    canales = []
    if espacio == "HSV":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        canales = ui.galeria({"H (tono)": hsv[..., 0], "S (saturación)": hsv[..., 1], "V (brillo)": hsv[..., 2]})
    return ui.rgb(img), fig, gr.Gallery(canales, visible=bool(canales))


def ecualizacion(*args):
    *vals, clip, baldosas = args
    img, _ = ui.resolver_imagen(*vals)
    glob = hi.ecualizar_global(img)
    clahe = hi.aplicar_clahe(img, float(clip), int(baldosas))
    salidas = []
    for titulo, im in [("Original", img), ("Global", glob), ("CLAHE", clahe)]:
        salidas.append(ui.figura(vis.grafica_histograma(hi.calcular_histograma(im, "Gris"), titulo)))
        salidas.append(ui.metricas({"Contraste RMS": f"{hi.contraste_rms(im):.1f}",
                                    "Desv. del brillo entre bloques": f"{hi.uniformidad_iluminacion(im):.1f}"}))
    gal = ui.galeria({"Original": img, "Ecualización global (V de HSV)": glob, "CLAHE (L de Lab)": clahe})
    return (gal, *salidas)


def uniformes(*args):
    *vals, fuente, metrica, nombre_det = args
    img, nombre = ui.resolver_imagen(*vals)
    if fuente == FUENTES_JUGADORES[0]:
        cajas = [r["caja"] for r in sg.segmentar_jugadores_clasico(img)["regiones"]]
    else:
        det = modelos.detectar(img, nombre_det, 0.6)
        cajas = [d["caja"] for d in det["detecciones"] if d["etiqueta"] == "person"]
    if len(cajas) < 2:
        return None, None, "⚠️ Se necesitan al menos 2 jugadores para comparar uniformes."
    hists = hi.histogramas_de_cajas(img, cajas)
    equipos, M = hi.asignar_equipos(hists, metrica)
    colores = {0: vis.ROJO, 1: vis.VERDE}
    anotada = img.copy()
    for i, (caja, e) in enumerate(zip(cajas, equipos)):
        anotada = vis.dibujar_cajas(anotada, [{"caja": caja, "etiqueta": f"{i + 1}:{'AB'[e]}", "color": colores[e]}])
        x1, y1, x2, y2 = hi.recorte_camiseta(caja)
        cv2.rectangle(anotada, (x1, y1), (x2, y2), vis.NEUTRO, 1)
    fig, ax = vis.figura(4.5, 4)
    im = ax.imshow(M, cmap="magma")
    ax.set_title(f"Matriz de {metrica}")
    ax.set_xticks(range(len(M)), [str(i + 1) for i in range(len(M))], fontsize=7)
    ax.set_yticks(range(len(M)), [str(i + 1) for i in range(len(M))], fontsize=7)
    fig.colorbar(im, ax=ax)
    texto = f"Equipo A: **{equipos.count(0)}** · Equipo B: **{equipos.count(1)}**"
    if ui.verdad_muestra(nombre) is not None and fuente == FUENTES_JUGADORES[0]:
        texto += "  \n<span class='vv-nota'>En la escena sintética el equipo A viste de rojo y el B de azul.</span>"
    return ui.rgb(anotada), ui.figura(fig), texto


def construir(estado: gr.State) -> None:
    ui.encabezado("3 · Histogramas", "Distribución de color, corrección de iluminación y comparación de uniformes.")
    ui.en_pocas_palabras("un histograma cuenta cuántos píxeles hay de cada tono. Sirve para corregir una foto muy "
                         "oscura o con luz desigual, y para comparar el color de las camisetas y saber quién es "
                         "de cada equipo.")
    sel = ui.selector_imagen()
    E = sel.entradas
    with gr.Tabs():
        with gr.Tab("Histograma por canal"):
            espacio = gr.Radio(["RGB", "HSV", "Gris"], value="RGB", label="Espacio de color")
            with gr.Row():
                original = ui.imagen_salida("Imagen")
                grafica = gr.Plot(show_label=False)
            canales = ui.nueva_galeria(3)
            ui.que_esta_pasando(r"""
El histograma cuenta cuántos píxeles tienen cada nivel: $h(r_k) = n_k$, y normalizado
$p(r_k) = n_k / N$. En **HSV** el tono H (0–179 en OpenCV) separa el color del brillo V: el pico
naranja de la cancha y el verde de la zona libre aparecen en H aunque la iluminación cambie.

Código: `core/histogramas.py` → `calcular_histograma` (usa `cv2.calcHist`).
""")
            gr.on(None, histograma, E + [espacio], [original, grafica, canales], trigger_mode="always_last")

        with gr.Tab("Ecualización y CLAHE"):
            clip = gr.Slider(0.5, 10, 2, step=0.5, label="CLAHE: intensidad de la corrección (clip limit)")
            with ui.opciones_avanzadas():
                baldosas = gr.Slider(2, 16, 8, step=1, label="CLAHE: baldosas por lado")
            gal = ui.nueva_galeria(3)
            salidas = []
            with gr.Row():
                for _ in range(3):
                    with gr.Column():
                        salidas.append(gr.Plot(show_label=False))
                        salidas.append(gr.HTML())
            ui.que_esta_pasando(r"""
**Ecualización global**: se usa la CDF como transformación,
$s_k = (L-1)\sum_{j\le k} p(r_j)$, para repartir los niveles por todo el rango. En color se aplica solo
a la luminancia para no alterar los tonos.

**CLAHE**: divide la imagen en baldosas, ecualiza cada una con su histograma **recortado** en el *clip
limit* (el exceso se reparte, así no se amplifica el ruido) e interpola entre baldosas. Corrige la
iluminación desigual del coliseo: compara la desviación del brillo entre bloques (menor = más pareja).

Código: `core/histogramas.py` → `ecualizar_manual` (CDF a mano), `ecualizar_global`, `aplicar_clahe`.
""")
            gr.on(None, ecualizacion, E + [clip, baldosas], [gal, *salidas], trigger_mode="always_last")

        with gr.Tab("Uniformes de los equipos"):
            with gr.Row():
                fuente = gr.Radio(list(FUENTES_JUGADORES), value=FUENTES_JUGADORES[0],
                                  label="¿Cómo encontrar a los jugadores?")
                nombre_det = gr.Dropdown(list(tm.MODELOS_DETECCION), value=next(iter(tm.MODELOS_DETECCION)),
                                         label="Detector transformer", visible=False)
            with ui.opciones_avanzadas():
                metrica = gr.Dropdown(list(hi.METODOS_COMPARACION), value=next(iter(hi.METODOS_COMPARACION)),
                                      label="Métrica de comparación")
            with gr.Row():
                with gr.Column(scale=3):
                    anotada = ui.imagen_salida("Equipo A (rojo) / equipo B (verde); recuadro blanco = torso usado")
                with gr.Column(scale=2):
                    matriz = gr.Plot(show_label=False)
                    conteo = ui.md()
            ui.que_esta_pasando(r"""
Para cada jugador se toma el **torso** (20–50 % de la altura de su caja) y se calcula el histograma 2D
de **tono y saturación** (H-S), ignorando píxeles grises u oscuros. Así no importa el brillo.

`cv2.compareHist` mide el parecido: **Correlación** (1 = iguales), **Chi-cuadrado** (0 = iguales),
**Intersección** $\sum\min(H_1,H_2)$ y **Bhattacharyya**
$d=\sqrt{1-\sum\sqrt{H_1H_2}/\sqrt{\bar H_1\bar H_2N^2}}$ (0 = iguales).

Asignación a equipos: se toman como referencia los dos jugadores **más distintos** y se itera
k-medias (k = 2) con esa métrica. Código: `core/histogramas.py` → `asignar_equipos`.
""")
            fuente.change(lambda f: gr.Dropdown(visible=f == FUENTES_JUGADORES[1]), fuente, nombre_det)
            gr.on(None, uniformes, E + [fuente, metrica, nombre_det], [anotada, matriz, conteo],
                  trigger_mode="always_last")
