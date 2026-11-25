"""9 · Pipeline completo."""

import gradio as gr
import pandas as pd

from volleyvision.core import pipeline as pl
from volleyvision.core import transformers_models as tm
from volleyvision.ui import componentes as ui
from volleyvision.ui import modelos
from volleyvision.utils import visualizacion as vis


def estado_esquinas(*args):
    *vals, usar_ejemplo, estado = args
    img, nombre = ui.resolver_imagen(*vals)
    if ui.esquinas_de(estado, nombre) is not None:
        return "Esquinas de la media cancha: ✅ marcadas"
    if usar_ejemplo and ui.esquinas_ejemplo(nombre, img) is not None:
        return "Esquinas de la media cancha: ✅ de ejemplo (guardadas con la foto)"
    return "Esquinas de la media cancha: ❌ sin marcar (el paso 7 se omite)"


def ejecutar(*args):
    """Ejecuta las 7 etapas; si se pide un transformer, la llamada usa la GPU (ZeroGPU)."""
    *vals, usar_calib, usar_ejemplo, clip, k, cierre, usar_det, nombre_det, umbral, usar_seg, estado = args
    img, nombre = ui.resolver_imagen(*vals)
    calib = (estado or {}).get("calibracion")
    if usar_calib and calib is None:
        gr.Warning("No hay calibración: calibra primero en la pestaña 1. Se sigue sin corregir la distorsión.")
    esquinas = ui.esquinas_de(estado, nombre)
    if esquinas is None and usar_ejemplo:
        esquinas = ui.esquinas_ejemplo(nombre, img)
    cfg = pl.ConfigPipeline(calibracion=calib if usar_calib else None, esquinas=esquinas, clip_limit=float(clip),
                            k_kmeans=int(k), tam_cierre=int(cierre), umbral_deteccion=float(umbral))
    if usar_det or usar_seg:
        pasos = modelos.pipeline_con_modelos(img, cfg, nombre_det if usar_det else None, bool(usar_seg))
    else:
        pasos = pl.ejecutar_pipeline(img, cfg)
    total = sum(p.tiempo_ms for p in pasos)
    fig = vis.grafica_barras([p.nombre.split(".")[0] for p in pasos], [p.tiempo_ms for p in pasos],
                             "Tiempo por etapa", ylabel="ms")
    return ui.metricas({"Tiempo total": f"{total:.0f} ms"}), ui.figura(fig), pasos


def _texto_valor(v) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    if isinstance(v, dict):
        return ", ".join(f"{a}: {b}" for a, b in v.items())
    return str(v)


def construir(estado: gr.State) -> None:
    ui.encabezado("9 · Pipeline completo", "Todas las etapas encadenadas sobre una imagen o un frame de video.")
    ui.en_pocas_palabras("es lo mismo que «Analizar partido», pero con todos los controles a la vista y el "
                         "resultado de cada una de las 7 etapas.")
    sel = ui.selector_imagen()
    E = sel.entradas
    with gr.Accordion("⚙️ Configuración", open=False):
        with gr.Row():
            with gr.Column():
                ui.md("**Calibración y cancha**")
                usar_calib = gr.Checkbox(False, label="Corregir distorsión con la calibración",
                                         info="Calibra primero en la pestaña 1. Solo para imágenes de esa cámara.")
                usar_ejemplo = gr.Checkbox(True, label="Usar las esquinas de ejemplo de la foto de muestra")
                texto_esq = ui.md()
            with gr.Column():
                ui.md("**Clásico**")
                clip = gr.Slider(0.5, 6, 2, step=0.5, label="CLAHE clip limit")
                k = gr.Slider(3, 8, 6, step=1, label="k del modelo de fondo")
                cierre = gr.Slider(3, 21, 9, step=2, label="Cierre vertical (px)")
            with gr.Column():
                ui.md("**Transformers**")
                usar_det = gr.Checkbox(True, label="Detector transformer")
                nombre_det = gr.Dropdown(list(tm.MODELOS_DETECCION), value=next(iter(tm.MODELOS_DETECCION)),
                                         label="Detector transformer")
                umbral = gr.Slider(0.1, 0.95, 0.5, step=0.05, label="Umbral de confianza")
                usar_seg = gr.Checkbox(False, label="SegFormer")
    with gr.Accordion("Marcar las esquinas de la media cancha", open=False):
        ui.selector_esquinas(sel, estado)
    boton = gr.Button("▶ Ejecutar pipeline", variant="primary")
    with gr.Row():
        total = gr.HTML(ui.metricas({"Tiempo total": "—"}))
        grafica = gr.Plot(show_label=False)
    pasos = gr.State(None)

    @gr.render(inputs=pasos)
    def mostrar_pasos(lista):
        if not lista:
            ui.md("Configura y pulsa **Ejecutar pipeline**.")
            return
        for paso in lista:
            ui.md(f"### {paso.nombre} <span class='vv-nota'>· {paso.tiempo_ms:.0f} ms</span>"
                  + (f"\n\n<span class='vv-nota'>{paso.nota}</span>" if paso.nota else ""))
            if paso.imagenes:
                ui.nueva_galeria(2 if len(paso.imagenes) in (2, 4) else 3, value=ui.galeria(paso.imagenes))
            escalares = {k: _texto_valor(v) for k, v in paso.datos.items() if not isinstance(v, list)}
            if escalares:
                gr.HTML(ui.metricas(escalares))
            for clave, v in paso.datos.items():
                if isinstance(v, list) and v:
                    with gr.Accordion(clave.replace("_", " ").capitalize(), open=False):
                        gr.Dataframe(pd.DataFrame(v), interactive=False)

    ui.que_esta_pasando(f"""
El pipeline (`core/pipeline.py` → `ejecutar_pipeline`) encadena los temas del curso:

1. **Calibración**: `cv2.undistort` con la K y la distorsión estimadas (las esquinas marcadas también se corrigen).
2. **Histograma**: CLAHE sobre L de Lab para emparejar la iluminación.
3. **Filtros**: bilateral + Canny + Hough para las líneas de la cancha.
4. **Umbralización + morfología**: distancia al fondo con Otsu, rangos HSV del balón, apertura y cierre.
5. **Segmentación**: contornos de la máscara clásica y, si se activa, {', '.join(tm.MODELOS_DETECCION)} / SegFormer.
6. **Caracterización**: descriptores, reglas "¿es balón?" y **solo tu equipo**: los que pisan tu media cancha
   (con las esquinas) o, sin ellas, el grupo de camisetas (histogramas H-S) más cercano a la cámara.
7. **Mapa cenital**: homografía (a partir de tu media cancha) aplicada al punto de apoyo de cada jugador;
   el mapa muestra tu media cancha, con la red arriba, como la ve la cámara trasera.

Si hay detector transformer y encuentra personas, el mapa usa sus cajas; si no, las del método clásico.
Con ZeroGPU, solo la ejecución con transformers pide GPU; la vía clásica corre en CPU.
""")
    gr.on(None, estado_esquinas, E + [usar_ejemplo, estado], texto_esq, trigger_mode="always_last")
    usar_det.change(lambda u: gr.Dropdown(visible=u), usar_det, nombre_det)
    boton.click(ejecutar, E + [usar_calib, usar_ejemplo, clip, k, cierre, usar_det, nombre_det, umbral, usar_seg,
                               estado], [total, grafica, pasos])
