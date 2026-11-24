"""Analizar partido: modo guiado para quien no es experto.

Tres pasos (imagen → esquinas opcionales → «Analizar») y un resultado en
lenguaje sencillo. Por debajo ejecuta el mismo pipeline de la pestaña 9, con
valores por defecto, y muestra cada etapa con el tema del curso que usa.
"""

import gradio as gr

from volleyvision.core import pipeline as pl
from volleyvision.core import transformers_models as tm
from volleyvision.ui import componentes as ui
from volleyvision.ui import modelos

DETECTOR = next(iter(tm.MODELOS_DETECCION))  # RT-DETR

# Explicación sencilla de cada etapa del pipeline y el tema del curso que usa.
EXPLICACION_PASOS = {
    "1": ("Corregir la lente", "Calibración",
          "Las cámaras deforman un poco la imagen (las líneas rectas se ven curvas). Si se calibró la cámara, "
          "se corrige aquí."),
    "2": ("Mejorar la luz", "Histogramas",
          "Se reparten mejor los tonos claros y oscuros para que la iluminación del coliseo sea más pareja."),
    "3": ("Encontrar las líneas de la cancha", "Filtros",
          "Se suaviza la imagen y se buscan los bordes; las líneas rectas largas son las de la cancha."),
    "4": ("Separar a los jugadores y el balón del piso", "Umbralización y morfología",
          "Se marca en blanco todo lo que no se parece al piso y se limpian los puntos sueltos."),
    "5": ("Encontrar a los jugadores", "Segmentación (clásica y con inteligencia artificial)",
          "Cada mancha blanca con forma de persona es un jugador. Si activaste la inteligencia artificial, "
          "un modelo entrenado con miles de fotos también los busca."),
    "6": ("Reconocer equipos y balón", "Caracterización",
          "Se compara el color de las camisetas para separar los dos equipos, y se busca una forma redonda "
          "amarilla para el balón."),
    "7": ("Ubicar a cada jugador en la cancha", "Calibración (homografía)",
          "Con las 4 esquinas se calcula cómo se vería la cancha desde arriba, y se coloca a cada jugador "
          "según dónde tiene los pies."),
}


def estado_esquinas(*args):
    """Texto del paso 2 según si la escena ya tiene esquinas."""
    *vals, estado = args
    img, nombre = ui.resolver_imagen(*vals)
    if ui.esquinas_de(estado, nombre) is not None:
        return "✅ Ya marcaste las 4 esquinas: el resultado incluirá el mapa de la cancha."
    if ui.esquinas_ejemplo(nombre, img) is not None:
        return "✅ Esta imagen de ejemplo ya trae sus esquinas: el resultado incluirá el mapa de la cancha."
    return ("ℹ️ Sin esquinas el análisis funciona igual, pero no podrá dibujar el mapa de la cancha. "
            "Si quieres el mapa, ábrelas aquí abajo y haz 4 clics.")


def _resumen(pasos, usar_ia: bool) -> str:
    datos6 = pasos[5].datos
    equipos = datos6.get("Jugadores por equipo", {"A": 0, "B": 0})
    total = equipos.get("A", 0) + equipos.get("B", 0)
    balon = any(f.get("es_balón") for f in datos6.get("tabla_balon", [])) or \
        pasos[4].datos.get("Balones (transformer)", 0) > 0
    lineas = [f"<b>Se encontraron {total} jugadores</b>: {equipos.get('A', 0)} de un equipo y "
              f"{equipos.get('B', 0)} del otro (cada equipo se pinta con el color de su camiseta).",
              "<b>Balón:</b> " + ("encontrado ✅" if balon else "no se encontró ❌")]
    posiciones = pasos[6].datos.get("posiciones_m")
    if posiciones:
        dentro = [p for p in posiciones if p["en cancha"]]
        propios = sum(p["y (m)"] > 9 for p in dentro)
        lineas.append(f"<b>En la cancha:</b> {propios} en tu lado (abajo en el mapa) y {len(dentro) - propios} "
                      "en el lado rival (arriba).")
    fuente = datos6.get("Fuente de jugadores")
    if usar_ia and fuente == "clásico":
        lineas.append("<span class='vv-nota'>La inteligencia artificial no reconoció personas en esta imagen "
                      "(pasa con dibujos como la escena de ejemplo), así que se usó el método clásico.</span>")
    tiempo = sum(p.tiempo_ms for p in pasos) / 1000
    lineas.append(f"<span class='vv-nota'>Análisis completo en {tiempo:.1f} s.</span>")
    return "<div class='vv-resumen'>" + "<br>".join(lineas) + "</div>"


def analizar(*args):
    """Ejecuta el pipeline con valores por defecto y devuelve un resultado fácil de leer."""
    *vals, usar_ia, estado = args
    img, nombre = ui.resolver_imagen(*vals)
    esquinas = ui.esquinas_de(estado, nombre)
    if esquinas is None:
        esquinas = ui.esquinas_ejemplo(nombre, img)
    cfg = pl.ConfigPipeline(esquinas=esquinas)
    if usar_ia:
        pasos = modelos.pipeline_con_modelos(img, cfg, DETECTOR, False)
    else:
        pasos = pl.ejecutar_pipeline(img, cfg)
    anotada = ui.rgb(pasos[5].imagenes["Equipos y balón"])
    mapa = pasos[6].imagenes.get("Mapa de posiciones")
    return anotada, (ui.rgb(mapa) if mapa is not None else None), _resumen(pasos, bool(usar_ia)), pasos


def construir(estado: gr.State) -> None:
    gr.Markdown("# 🏐 Volley<span class='acento'>Vision</span>\n"
                "Sube una foto o un video de un partido de voleibol tomado **desde detrás de tu equipo** y la "
                "app encuentra a los jugadores, los separa por equipo, busca el balón y los ubica en un mapa "
                "de la cancha.")
    gr.HTML("<div class='vv-paso'><span class='num'>1</span>Elige la imagen</div>")
    sel = ui.selector_imagen()
    E = sel.entradas

    gr.HTML("<div class='vv-paso'><span class='num'>2</span>Marca las esquinas de tu media cancha "
            "<span class='vv-nota'>(opcional)</span></div>")
    texto_esq = ui.md()
    with gr.Accordion("Marcar las 4 esquinas con clics", open=False):
        ui.md("Haz clic en las 4 esquinas de **tu** mitad de la cancha: primero las dos que están bajo la "
              "red (izquierda y derecha) y después las dos de tu línea de fondo (derecha e izquierda).")
        ui.selector_esquinas(sel, estado)

    gr.HTML("<div class='vv-paso'><span class='num'>3</span>Analiza</div>")
    usar_ia = gr.Checkbox(True, label="Usar también inteligencia artificial (RT-DETR) para encontrar a los "
                                      "jugadores: más preciso en fotos reales")
    boton = gr.Button("▶ Analizar", variant="primary", elem_id="vv-analizar")

    resumen = gr.HTML()
    with gr.Row():
        anotada = ui.imagen_salida("Jugadores por equipo y balón")
        mapa = ui.imagen_salida("Mapa de la cancha visto desde arriba")
    pasos = gr.State(None)

    @gr.render(inputs=pasos)
    def como_se_hizo(lista):
        if not lista:
            return
        with gr.Accordion("🔍 ¿Cómo se hizo? Paso a paso, con el tema del curso que usa cada uno", open=False):
            for paso in lista:
                titulo, tema, texto = EXPLICACION_PASOS[paso.nombre.split(".")[0]]
                ui.md(f"**{paso.nombre.split('.')[0]}. {titulo}** · <span class='acento'>{tema}</span>\n\n{texto}")
                if paso.imagenes:
                    ui.nueva_galeria(min(4, len(paso.imagenes)), value=ui.galeria(paso.imagenes))
            ui.md("Para explorar cada técnica con sus controles y su teoría, abre **📚 Temas del curso**.")

    gr.on(None, estado_esquinas, E + [estado], texto_esq, trigger_mode="always_last")
    boton.click(analizar, E + [usar_ia, estado], [anotada, mapa, resumen, pasos])
