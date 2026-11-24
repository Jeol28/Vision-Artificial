import numpy as np
import streamlit as st

from volleyvision.core import calibracion as cal
from volleyvision.ui import componentes as ui
from volleyvision.utils import io, sinteticos
from volleyvision.utils import visualizacion as vis

ui.encabezado("1 · Modelos de cámara y calibración",
              "Calibración intrínseca con tablero de ajedrez y homografía de la cancha desde la cámara trasera.")

tab_intr, tab_hom = st.tabs(["Calibración intrínseca", "Homografía de la cancha"])

# ---------------------------------------------------------------------------
with tab_intr:
    c1, c2, c3, c4 = st.columns(4)
    fuente = c1.radio("Fotos del tablero", ["Tablero sintético (ejemplo)", "Subir fotos"], key="cal_fuente")
    cols = c2.number_input("Esquinas interiores (columnas)", 3, 20, sinteticos.PATRON[0], key="cal_cols")
    filas = c3.number_input("Esquinas interiores (filas)", 3, 20, sinteticos.PATRON[1], key="cal_filas")
    cuadro = c4.number_input("Lado del cuadro (mm)", 1.0, 200.0, sinteticos.TAM_CUADRO_MM, key="cal_mm")
    patron = (int(cols), int(filas))

    if fuente == "Subir fotos":
        archivos = st.file_uploader("Fotos del tablero (mínimo 3, ideal 10–20 desde ángulos distintos)",
                                    type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="cal_archivos")
        imagenes = [io.cargar_imagen(a.getvalue()) for a in archivos or []]
    else:
        imagenes = [io.cargar_imagen(p) for p in io.listar_tableros()]
        st.caption(f"{len(imagenes)} vistas sintéticas generadas con K y distorsión conocidas "
                   f"(k₁ = {sinteticos.DIST_VERDADERA[0]}, k₂ = {sinteticos.DIST_VERDADERA[1]}).")

    if st.button("Calibrar cámara", type="primary", disabled=len(imagenes) < 3):
        try:
            with st.spinner("Detectando esquinas y optimizando…"):
                st.session_state["calibracion"] = cal.calibrar_camara(imagenes, patron, cuadro)
                st.session_state["calibracion_imagenes"] = imagenes
                st.session_state["calibracion_patron"] = patron
        except ValueError as e:
            st.error(str(e))

    res: cal.ResultadoCalibracion | None = st.session_state.get("calibracion")
    if res is None:
        st.info("Pulsa **Calibrar cámara** para estimar K y los coeficientes de distorsión.")
    else:
        imgs = st.session_state["calibracion_imagenes"]
        patron_res = st.session_state["calibracion_patron"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Error de reproyección (RMS)", f"{res.error_rms:.3f} px")
        m2.metric("Vistas válidas", f"{len(res.indices_validos)} / {len(imgs)}")
        m3.metric("Tamaño de imagen", f"{res.tam_imagen[0]}×{res.tam_imagen[1]}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Matriz intrínseca K**")
            st.latex(r"K = \begin{bmatrix} %.1f & %.1f & %.1f \\ 0 & %.1f & %.1f \\ 0 & 0 & 1 \end{bmatrix}"
                     % (res.K[0, 0], res.K[0, 1], res.K[0, 2], res.K[1, 1], res.K[1, 2]))
            if fuente != "Subir fotos":
                Kv = sinteticos.K_VERDADERA
                st.caption(f"Verdadera: fx = {Kv[0, 0]:.0f}, fy = {Kv[1, 1]:.0f}, "
                           f"cx = {Kv[0, 2]:.0f}, cy = {Kv[1, 2]:.0f}")
            st.markdown("**Coeficientes de distorsión** (k₁, k₂, p₁, p₂, k₃)")
            st.code(np.array2string(res.dist, precision=4, suppress_small=True))
        with c2:
            ui.figura(vis.grafica_barras([str(i) for i in res.indices_validos], res.errores_por_imagen,
                                         "Error de reproyección por vista", ylabel="px"))

        st.markdown("#### Esquinas detectadas")
        muestra = {f"Vista {i}": cal.dibujar_esquinas(imgs[i], patron_res, cal.encontrar_esquinas(imgs[i], patron_res)[1])
                   for i in res.indices_validos[:3]}
        ui.cuadricula(muestra, 3)

        st.markdown("#### Corrección de distorsión")
        a, b = st.columns(2)
        vista = a.selectbox("Vista", res.indices_validos, key="cal_vista")
        alfa = b.slider("α (0 = recortar, 1 = conservar todo)", 0.0, 1.0, 0.0, 0.05, key="cal_alfa")
        corregida, _ = cal.corregir_distorsion(imgs[vista], res.K, res.dist, alfa)
        ui.lado_a_lado(imgs[vista], corregida, "Con distorsión (rectas curvadas)", "Corregida (cv2.undistort)")

    ui.que_esta_pasando(r"""
**Modelo pinhole.** Un punto 3D se proyecta como $s\,[u, v, 1]^\top = K\,[R \mid t]\,[X, Y, Z, 1]^\top$, con

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

($f_x, f_y$ focales en píxeles, $(c_x, c_y)$ punto principal) y $[R \mid t]$ la pose de la cámara.

**Distorsión radial** (barril si $k_1 < 0$): $x_d = x\,(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + \dots$

**Método de Zhang.** Cada foto del tablero (plano $Z=0$) da una homografía $H = K[r_1\ r_2\ t]$; con
≥ 3 vistas se despeja $K$ y luego se refina todo minimizando el **error de reproyección**
$E = \sqrt{\tfrac{1}{N}\sum \lVert p_i - \hat p_i \rVert^2}$. Un error < 0.5 px es una buena calibración.

Código: `core/calibracion.py` → `calibrar_camara`, `corregir_distorsion`.
""")

# ---------------------------------------------------------------------------
with tab_hom:
    img, nombre = ui.selector_imagen("hom")
    c1, c2 = st.columns([3, 2])
    with c1:
        esquinas = ui.selector_esquinas(img, nombre, "hom")
    with c2:
        px_m = st.slider("Resolución de la vista cenital (px/m)", 10, 60, 30, key="hom_ppm")
        margen = st.slider("Zona libre alrededor (m)", 0.0, 5.0, 2.0, 0.5, key="hom_margen")
        if esquinas is not None:
            hom = cal.homografia_cancha(esquinas, px_m, margen)
            ui.mostrar(cal.vista_cenital(img, hom), "Vista cenital (cv2.warpPerspective)")
            st.markdown("**Homografía imagen → metros**")
            st.code(np.array2string(hom.H_metros, precision=5, suppress_small=True))
            H_dlt = cal.homografia_dlt(esquinas, cal.ESQUINAS_MEDIA_CANCHA_M)
            st.caption(f"DLT propio vs OpenCV: diferencia máx. = {np.abs(H_dlt - hom.H_metros / hom.H_metros[2, 2]).max():.2e}")
        else:
            st.info("Marca las 4 esquinas de tu media cancha para ver la cancha desde arriba. "
                    "Quedan guardadas para las páginas de Caracterización y Pipeline.")

    ui.que_esta_pasando(r"""
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
""")
