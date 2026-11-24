"""5 · Morfología matemática."""

import gradio as gr
import numpy as np

from volleyvision.core import caracterizacion as ca
from volleyvision.core import filtros as fi
from volleyvision.core import morfologia as mo
from volleyvision.core import segmentacion as sg
from volleyvision.core import umbralizacion as um
from volleyvision.ui import componentes as ui

MASCARAS = ("Jugadores (distancia al fondo + Otsu)", "Balón (HSV amarillo + azul)", "Otsu de la imagen en grises")


def mascara_entrada(img: np.ndarray, fuente: str, ruido: float) -> np.ndarray:
    if fuente == MASCARAS[0]:
        mascara = sg.mascara_primer_plano_kmeans(img)[0]
        mascara[sg.mascara_lineas_blancas(img) > 0] = 0
    elif fuente == MASCARAS[1]:
        mascara = um.mascara_por_nombre(img, "Balón (amarillo)") | um.mascara_por_nombre(img, "Balón (azul)")
    else:
        mascara = um.otsu_opencv(img)[1]
    if ruido > 0:
        mascara = fi.agregar_ruido(mascara, "sal_pimienta", ruido / 100, semilla=1)
    return mascara


def morfologia(*args):
    *vals, fuente, ruido, operacion, forma, tam, iters, ap, ci, area = args
    img, _ = ui.resolver_imagen(*vals)
    tam = int(tam)
    mascara = mascara_entrada(img, fuente, float(ruido))
    ee = mo.elemento_estructurante(forma, tam)
    resultado = mo.aplicar(mascara, operacion, ee, int(iters))

    entrada = gr.Image(ui.rgb(mascara), label=f"Entrada: {mo.contar_componentes(mascara)} componentes")
    salida = gr.Image(ui.rgb(resultado), label=f"{operacion}: {mo.contar_componentes(resultado)} componentes")
    ee_img = gr.Image(ui.rgb(np.kron(ee * 255, np.ones((12, 12), np.uint8))),
                      label=f"Elemento estructurante {tam}×{tam}")
    ee_txt = ("```\n" + "\n".join(" ".join(str(v) for v in fila) for fila in ee) + "\n```") if tam <= 9 \
        else f"{tam}×{tam}"
    todas = ui.galeria({op: mo.aplicar(mascara, op, ee, 1) for op in mo.OPERACIONES[:6]})

    recorte = mascara[: min(240, mascara.shape[0]), : min(320, mascara.shape[1])]
    iguales_e = np.array_equal(mo.erosion_manual(recorte, ee), mo.erosion(recorte, ee))
    iguales_d = np.array_equal(mo.dilatacion_manual(recorte, ee), mo.dilatacion(recorte, ee))
    comparacion = (f"Erosión manual = cv2.erode: **{'sí' if iguales_e else 'no'}** · "
                   f"Dilatación manual = cv2.dilate: **{'sí' if iguales_d else 'no'}**")

    es_balon = fuente == MASCARAS[1]
    limpia = mo.limpiar_mascara(mascara, int(ap), int(ci), forma, int(area), rellenar=es_balon)
    antes = ui.rgb(mascara)
    despues = gr.Image(ui.rgb(limpia), label=f"Después: {mo.contar_componentes(limpia)} componentes")
    nota = ""
    cands = ca.detectar_balon_clasico(img) if es_balon else []
    if cands:
        nota = (f"<span class='vv-nota'>Con la máscara limpia, el clasificador de la pestaña 7 encuentra "
                f"{sum(c['es_balon'] for c in cands)} balón(es) entre {len(cands)} candidatos.</span>")
    return entrada, salida, ee_img, ee_txt, todas, comparacion, antes, despues, nota


def construir(estado: gr.State) -> None:
    ui.encabezado("5 · Morfología matemática", "Limpiar las máscaras del balón y de los jugadores.")
    sel = ui.selector_imagen()
    with gr.Row():
        fuente = gr.Dropdown(list(MASCARAS), value=MASCARAS[0], label="Máscara de entrada")
        ruido = gr.Slider(0, 10, 1, step=0.5, label="Ruido sal y pimienta extra (%)")
    with gr.Row():
        operacion = gr.Dropdown(list(mo.OPERACIONES), value=mo.OPERACIONES[2], label="Operación")
        forma = gr.Dropdown(list(mo.FORMAS), value=list(mo.FORMAS)[1], label="Forma del elemento estructurante")
        tam = gr.Slider(1, 31, 5, step=2, label="Tamaño")
        iters = gr.Slider(1, 5, 1, step=1, label="Iteraciones")
    with gr.Row():
        with gr.Column(scale=2):
            entrada = ui.imagen_salida("Entrada")
        with gr.Column(scale=2):
            salida = ui.imagen_salida("Resultado")
        with gr.Column(scale=1):
            ee_img = ui.imagen_salida("Elemento estructurante")
            ee_txt = ui.md()
    ui.md("#### Todas las operaciones con el mismo elemento estructurante")
    todas = ui.nueva_galeria(3)
    ui.md("#### Implementación manual vs OpenCV")
    comparacion = ui.md()
    ui.md("#### Limpieza completa (apertura → cierre → relleno → filtro de área)")
    with gr.Row():
        ap = gr.Slider(1, 15, 3, step=2, label="Apertura")
        ci = gr.Slider(1, 21, 7, step=2, label="Cierre")
        area = gr.Slider(0, 500, 40, step=10, label="Área mínima (px)")
    with gr.Row():
        antes = ui.imagen_salida("Antes")
        despues = ui.imagen_salida("Después")
    nota = ui.md()
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
    gr.on(None, morfologia, sel.entradas + [fuente, ruido, operacion, forma, tam, iters, ap, ci, area],
          [entrada, salida, ee_img, ee_txt, todas, comparacion, antes, despues, nota], trigger_mode="always_last")
