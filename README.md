---
title: VolleyVision
emoji: 🏐
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Pipeline de visión por computador para voleibol
---

# 🏐 VolleyVision

**Proyecto final – Visión Artificial** · Ingeniería de Sistemas, Pontificia Universidad Javeriana
**Autor:** Jorge Enrique Olaya Liévano

> **App en línea:** [huggingface.co/spaces/Jeol28/Vision](https://huggingface.co/spaces/Jeol28/Vision) (Hugging Face Spaces con ZeroGPU). Si el Space estaba dormido, la primera carga tarda un par de minutos.

VolleyVision analiza fotos y video de un partido de voleibol con un **pipeline de visión por computador**.
La cámara se ubica **detrás de la línea de fondo del equipo propio** (vista trasera), de modo que se
ve al equipo propio de espaldas, con el número de la camiseta visible, y al rival de frente al otro lado
de la red. **Solo se analiza el equipo propio**: desde atrás el rival queda lejos, pequeño y tapado por
la red, así que se descarta. Cada tema del curso es una etapa del pipeline: calibración de la cámara, filtros, histogramas,
umbralización, morfología matemática, segmentación y caracterización. Al final se comparan las técnicas
clásicas con **modelos transformer** de Hugging Face (RT-DETR, DETR, SegFormer y SAM).

> No se entrena ni se etiqueta ningún modelo: los transformers se usan **solo en inferencia** con
> pesos públicos que se descargan bajo demanda.

---

## Contenido

- [Capturas](#capturas)
- [Instalación](#instalación)
- [Cómo ejecutar](#cómo-ejecutar)
- [Despliegue en Hugging Face Spaces (ZeroGPU)](#despliegue-en-hugging-face-spaces-zerogpu)
- [Estrategia de ramificación](#estrategia-de-ramificación)
- [Arquitectura](#arquitectura)
- [Módulos](#módulos)
- [Modelos transformer](#modelos-transformer)
- [Pruebas](#pruebas)
- [Datos de ejemplo](#datos-de-ejemplo)
- [Limitaciones](#limitaciones)
- [Licencia](#licencia)

---

## Capturas

La de Transformers se tomó del Space desplegado (los tiempos de RT-DETR y DETR son en la GPU de
ZeroGPU); las demás, de la app local con el mismo código.

| Analizar partido (modo guiado) | Calibración y homografía |
|---|---|
| ![Analizar partido: resultado en lenguaje sencillo, tu equipo numerado y el mapa de tu media cancha](docs/capturas/analizar.png) | ![Homografía de la media cancha y vista cenital](docs/capturas/calibracion.png) |
| **Transformers (clásico vs. RT-DETR vs. DETR)** | **Pipeline completo (mapa cenital)** |
| ![Comparación clásico vs. transformer](docs/capturas/transformers.png) | ![Pipeline: paso 7, mapa cenital](docs/capturas/pipeline.png) |

---

## Instalación

Requisitos: **Python 3.11 o superior** (probado con 3.11 en local y 3.12 en el Space) y ~2 GB libres para los pesos de los modelos (se descargan la primera vez
que se usan, en la caché de Hugging Face, no en el repositorio).

```bash
git clone <url-del-repositorio> VolleyVision
cd VolleyVision

python3 -m venv .venv               # Python 3.11 o superior
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Funciona en **CPU**. Si hay GPU la usa automáticamente (CUDA en NVIDIA, MPS en Apple Silicon). Para
forzar un dispositivo: `export VOLLEYVISION_DEVICE=cpu`.

Datos de ejemplo (opcional):

```bash
python scripts/generar_sinteticos.py   # ya vienen en el repo; regenera tablero, cancha y video sintéticos
python scripts/download_samples.py     # descarga fotos reales con licencia libre de Wikimedia Commons
```

## Cómo ejecutar

```bash
python app.py
```

Se abre en `http://127.0.0.1:7860`. La interfaz está hecha con **Gradio** y pensada también para quien no
sabe de visión artificial. Tiene tres secciones:

| Sección | Para qué sirve |
|---|---|
| 🏐 **Analizar partido** | Modo guiado en 3 pasos: elegir la imagen → marcar las 4 esquinas de la media cancha (opcional) → **Analizar**. El resultado sale en lenguaje sencillo (cuántos jugadores de tu equipo hay, si está el balón y cuántos están en la zona de ataque o de defensa) junto con la foto anotada y el mapa de tu media cancha visto desde arriba. El rival se ignora. Un desplegable *¿Cómo se hizo?* muestra las 7 etapas y el tema del curso que usa cada una. |
| 📚 **Temas del curso** | Una pestaña por tema (1 · Calibración … 9 · Pipeline). Cada una empieza con un recuadro **"En pocas palabras"**, deja a la vista solo los controles principales, guarda el resto en **⚙️ Opciones avanzadas** y tiene la teoría con fórmulas en **"¿Qué está pasando?"**. |
| ℹ️ **Acerca del proyecto** | Autor, recorrido, pipeline y arquitectura. |

Las esquinas marcadas y la calibración se guardan en la sesión y las comparten todas las secciones.

## Despliegue en Hugging Face Spaces (ZeroGPU)

La app está preparada para un Space de Gradio con **ZeroGPU** (GPU compartida que se asigna solo
mientras corre una inferencia). La cabecera YAML al inicio de este README es la configuración del Space.

1. Crear el Space en <https://huggingface.co/new-space>: SDK **Gradio**, hardware **ZeroGPU**
   (requiere una cuenta PRO o de organización).
2. Subir el repositorio (la herramienta `hf` maneja sola los archivos binarios: JPG, PNG y MP4):

   ```bash
   pip install -U huggingface_hub
   hf auth login
   hf upload <usuario>/volleyvision . . --repo-type=space \
       --exclude ".venv/*" ".git/*" "**/__pycache__/*" ".pytest_cache/*" ".gradio/*"
   ```

Cómo funciona ZeroGPU en el código (`volleyvision/ui/modelos.py`):

- `app.py` importa `spaces` antes que torch y, si detecta ZeroGPU (`SPACES_ZERO_GPU`), **carga los 5
  modelos al arrancar** y los mueve a `cuda`, como exige ZeroGPU.
- La inferencia está en funciones con `@spaces.GPU` (`detectar`, `segmentar_semantica`,
  `segmentar_con_sam`, `comparar`, `pipeline_con_modelos`). Solo esas piden GPU; las técnicas clásicas
  corren en CPU y no gastan cuota.
- Cada llamada puede caer en una GPU distinta, así que los detectores se calientan dentro de la llamada
  antes de medir el tiempo.
- Fuera de ZeroGPU, `@spaces.GPU` no hace nada y los modelos se cargan la primera vez que se usan.
- Cada función reserva poca GPU (15–30 s): ZeroGPU descuenta la reserva de la cuota del visitante
  antes de ejecutar, y con reservas grandes un visitante sin sesión se queda sin cuota enseguida.

Los visitantes tienen una cuota diaria de GPU (mayor si inician sesión en Hugging Face). Si el Space
lleva tiempo sin visitas se duerme y la primera visita tarda en arrancar.

---

## Estrategia de ramificación

Se usa un flujo tipo **Git Flow simplificado**:

| Rama | Propósito |
|---|---|
| `main` | Versión estable y entregable. Solo recibe merges desde `develop` y se etiqueta en cada versión (`v1.0.0`, `v1.1.0`, `v2.0.0`, `v2.0.1`, `v2.1.0`, `v2.2.0`). |
| `develop` | Rama de integración: aquí se juntan los módulos terminados y probados. |
| `feature/<tema>` | Una rama por módulo del curso: `feature/calibracion`, `feature/filtros`, `feature/histogramas`, `feature/umbralizacion`, `feature/morfologia`, `feature/segmentacion`, `feature/caracterizacion`, `feature/transformers`, `feature/ui` (y `feature/docs` para la documentación). Los cambios posteriores siguen el mismo flujo: `feature/camara-trasera` cambió la cámara a la vista trasera (`v1.1.0`) y `feature/gradio-zerogpu` pasó la interfaz a Gradio con ZeroGPU (`v2.0.0`); `feature/ui-sencilla` agregó el modo guiado para no expertos (`v2.1.0`) y `feature/solo-mi-equipo` hizo que solo se analice el equipo propio (`v2.2.0`). |

Reglas:

1. Cada `feature/*` nace de `develop` y vuelve a `develop` con `git merge --no-ff`, para que el
   historial conserve la rama como una unidad.
2. Commits pequeños con [Conventional Commits](https://www.conventionalcommits.org/es/):
   `feat(filtros): …`, `test(umbralizacion): …`, `fix(transformers): …`, `docs: …`, `chore: …`.
3. Antes de cada merge se ejecuta `pytest` y la rama debe pasar todas las pruebas.
4. Cuando `develop` está estable se integra a `main` y se crea la etiqueta de versión.

```mermaid
gitGraph
    commit id: "chore: inicializar repo"
    branch develop
    checkout develop
    commit id: "chore: estructura en capas"
    commit id: "feat(utils): sintéticos"
    branch feature/calibracion
    checkout feature/calibracion
    commit id: "feat(calibracion)"
    commit id: "test(calibracion)"
    checkout develop
    merge feature/calibracion
    branch feature/filtros
    checkout feature/filtros
    commit id: "feat(filtros)"
    commit id: "test(filtros)"
    checkout develop
    merge feature/filtros
    branch feature/histogramas
    checkout feature/histogramas
    commit id: "feat(histogramas)"
    commit id: "test(histogramas)"
    checkout develop
    merge feature/histogramas
    branch feature/umbralizacion
    checkout feature/umbralizacion
    commit id: "feat(umbralizacion)"
    commit id: "test(umbralizacion)"
    checkout develop
    merge feature/umbralizacion
    branch feature/morfologia
    checkout feature/morfologia
    commit id: "feat(morfologia)"
    commit id: "test(morfologia)"
    checkout develop
    merge feature/morfologia
    branch feature/segmentacion
    checkout feature/segmentacion
    commit id: "feat(segmentacion)"
    commit id: "test(segmentacion)"
    checkout develop
    merge feature/segmentacion
    branch feature/caracterizacion
    checkout feature/caracterizacion
    commit id: "feat(caracterizacion)"
    commit id: "test(caracterizacion)"
    checkout develop
    merge feature/caracterizacion
    branch feature/transformers
    checkout feature/transformers
    commit id: "feat(transformers)"
    commit id: "feat(pipeline)"
    commit id: "test(transformers)"
    checkout develop
    merge feature/transformers
    branch feature/ui
    checkout feature/ui
    commit id: "feat(ui): componentes"
    commit id: "feat(ui): páginas"
    commit id: "test(ui)"
    checkout develop
    merge feature/ui
    branch feature/docs
    checkout feature/docs
    commit id: "docs: README y guion"
    checkout develop
    merge feature/docs
    checkout main
    merge develop tag: "v1.0.0"
    checkout develop
    branch feature/camara-trasera
    checkout feature/camara-trasera
    commit id: "feat: cámara trasera"
    commit id: "docs: cámara trasera"
    checkout develop
    merge feature/camara-trasera
    checkout main
    merge develop tag: "v1.1.0"
    checkout develop
    branch feature/gradio-zerogpu
    checkout feature/gradio-zerogpu
    commit id: "feat(ui): Gradio + ZeroGPU"
    commit id: "docs: Gradio y despliegue"
    checkout develop
    merge feature/gradio-zerogpu
    checkout main
    merge develop tag: "v2.0.0"
    checkout develop
    branch feature/docs-entrega
    checkout feature/docs-entrega
    commit id: "fix(ui): cuota ZeroGPU"
    commit id: "docs: capturas y enlace"
    checkout develop
    merge feature/docs-entrega
    checkout main
    merge develop tag: "v2.0.1"
    checkout develop
    branch feature/ui-sencilla
    checkout feature/ui-sencilla
    commit id: "feat(ui): modo guiado"
    commit id: "docs: modo guiado"
    checkout develop
    merge feature/ui-sencilla
    checkout main
    merge develop tag: "v2.1.0"
    checkout develop
    branch feature/solo-mi-equipo
    checkout feature/solo-mi-equipo
    commit id: "feat: analizar solo el equipo propio"
    commit id: "test: equipo propio"
    checkout develop
    merge feature/solo-mi-equipo
    checkout main
    merge develop tag: "v2.2.0"
```

Para ver el historial real: `git log --oneline --graph --all`.

---

## Arquitectura

### Arquitectura en capas

El código se organiza en tres capas con dependencias en un solo sentido
(**UI → core → utils**):

```mermaid
flowchart TB
    subgraph UI["Capa de presentación · volleyvision/ui (Gradio)"]
        direction LR
        APP["app.py<br/>secciones y estado"] ~~~ PAG["paginas/<br/>analizar (guiado), p0 … p9"] ~~~ COMP["componentes.py<br/>imagen/video, esquinas por clic"] ~~~ MOD["modelos.py<br/>carga única + @spaces.GPU"]
    end
    subgraph CORE["Capa de dominio · volleyvision/core (funciones puras, sin Gradio)"]
        direction LR
        C1["calibracion<br/>filtros<br/>histogramas"] ~~~ C2["umbralizacion<br/>morfologia<br/>segmentacion"] ~~~ C3["caracterizacion<br/>transformers_models<br/>pipeline"]
    end
    subgraph UTILS["Capa de utilidades · volleyvision/utils"]
        direction LR
        U1["io<br/>imágenes y video"] ~~~ U2["visualizacion<br/>dibujo y gráficas"] ~~~ U3["sinteticos<br/>datos con verdad conocida"]
    end
    subgraph EXT["Dependencias externas"]
        direction LR
        E1["OpenCV · NumPy<br/>scikit-image · Matplotlib"] ~~~ E2["PyTorch<br/>HF transformers"] ~~~ E3[("Hugging Face Hub<br/>pesos públicos")]
    end
    UI -->|"solo llama al core"| CORE
    CORE -->|"usa"| UTILS
    CORE --> EXT
```

- **`core/`**: un módulo por tema del curso. Todas las funciones son **puras**: reciben una imagen
  (arreglo NumPy BGR `uint8`) y parámetros, y devuelven imágenes y/o datos. No importan Gradio, así
  que se prueban con `pytest` y se pueden usar desde un script o un cuaderno.
- **`ui/`**: Gradio. Solo recoge parámetros con widgets, llama al core y muestra resultados. Los
  modelos se cargan **una vez** por proceso (`ui/modelos.py`) y la inferencia va en funciones
  `@spaces.GPU` para ZeroGPU.
- **`utils/`**: E/S de imágenes y video, dibujo con la paleta del proyecto y generación de datos
  sintéticos con verdad conocida (para pruebas y demos sin internet).

### Pipeline

```mermaid
flowchart LR
    A["Imagen o<br/>frame de video"] --> B["1 · Corrección de<br/>distorsión<br/><i>calibracion</i>"]
    B --> C["2 · CLAHE<br/><i>histogramas</i>"]
    C --> D["3 · Bilateral + Canny<br/>+ Hough<br/><i>filtros</i>"]
    D --> E["4 · Otsu / HSV<br/>+ apertura y cierre<br/><i>umbralizacion, morfologia</i>"]
    E --> F1["5a · Segmentación clásica<br/>k-means, contornos, watershed"]
    E --> F2["5b · Segmentación transformer<br/>RT-DETR / DETR, SegFormer"]
    F1 --> G["6 · Caracterización<br/>descriptores, ¿es balón?,<br/>solo tu equipo"]
    F2 --> G
    G --> H["7 · Mapa cenital<br/>homografía de la cancha"]
```

Cada etapa devuelve un objeto `Paso` con sus imágenes intermedias, datos y tiempo, y la pestaña
**Pipeline** los muestra todos. Más detalle en [docs/arquitectura.md](docs/arquitectura.md).

### Estructura del repositorio

```
VolleyVision/
├── volleyvision/
│   ├── core/                 # lógica de visión (funciones puras)
│   │   ├── calibracion.py    ├── filtros.py         ├── histogramas.py
│   │   ├── umbralizacion.py  ├── morfologia.py      ├── segmentacion.py
│   │   ├── caracterizacion.py├── transformers_models.py
│   │   └── pipeline.py
│   ├── ui/                   # Gradio
│   │   ├── app.py  ├── componentes.py  ├── modelos.py
│   │   └── paginas/  analizar.py (modo guiado), p0_inicio.py … p9_pipeline.py
│   └── utils/                # io.py, visualizacion.py, sinteticos.py
├── data/samples/             # cancha y video sintéticos, tablero (descargadas/ se ignora)
├── scripts/                  # generar_sinteticos.py, download_samples.py
├── tests/                    # pytest (103 pruebas)
├── docs/                     # arquitectura, guion del video, capturas
├── app.py                    # punto de entrada (local y Hugging Face Spaces)
├── requirements.txt  ├── pytest.ini  ├── LICENSE  └── README.md
```

---

## Módulos

| Módulo | Qué hace (aplicado al voleibol) | Funciones clave |
|---|---|---|
| `calibracion` | Calibración de Zhang con tablero de ajedrez (K, distorsión, error de reproyección), corrección de distorsión y homografía de la cancha desde la **cámara trasera**: con las 4 esquinas de la media cancha propia (9 × 9 m) se obtiene la homografía de todo el piso y la vista cenital de la cancha de 9 × 18 m. Modelo pinhole explicado en los docstrings. Incluye DLT propia comparada con OpenCV. | `calibrar_camara`, `corregir_distorsion`, `homografia_cancha`, `homografia_dlt`, `vista_cenital` |
| `filtros` | Promedio, gaussiano, mediana y bilateral; Sobel, Laplaciano y Canny; convolución manual (NumPy) comparada con `cv2.filter2D`; realce de las líneas de la cancha (bilateral + Canny + Hough). | `suavizar`, `sobel`, `canny`, `convolucion_manual`, `resaltar_lineas_cancha` |
| `histogramas` | Histogramas RGB/HSV, ecualización global (y manual con la CDF), CLAHE para la iluminación del coliseo y comparación H-S de uniformes con `cv2.compareHist` para separar tu equipo del rival. | `calcular_histograma`, `ecualizar_manual`, `aplicar_clahe`, `asignar_equipos` |
| `umbralizacion` | Umbral global, **Otsu implementado a mano** con la curva σ²_B(t), Otsu de OpenCV, umbral adaptativo (media y gaussiano) y rangos HSV para el balón y la cancha. | `otsu_manual`, `umbral_adaptativo`, `umbral_hsv` |
| `morfologia` | Erosión, dilatación, apertura, cierre, gradiente, top-hat y black-hat con elemento estructurante configurable; versiones manuales (mín/máx); limpieza de las máscaras del balón y los jugadores. | `elemento_estructurante`, `aplicar`, `erosion_manual`, `limpiar_mascara` |
| `segmentacion` | Contornos, watershed con transformada de distancia, k-means en color y detección clásica de jugadores (modelo de fondo + Otsu + cierre vertical). Métricas IoU. | `encontrar_contornos`, `watershed_distancia`, `kmeans_color`, `segmentar_jugadores_clasico` |
| `caracterizacion` | Área, perímetro, centroide, caja, circularidad, excentricidad, solidez y momentos de Hu; clasificador por reglas "¿es balón?"; proyección del punto de apoyo de los jugadores a la vista cenital; selección del **equipo propio** (los que pisan tu media cancha o, sin esquinas, el grupo de camisetas más cercano a la cámara). | `describir_region`, `detectar_balon_clasico`, `proyectar_jugadores`, `seleccionar_equipo_propio` |
| `transformers_models` | RT-DETR / DETR (personas y `sports ball`), SegFormer (segmentación semántica), SlimSAM / SAM (segmentación por punto o caja) y la comparación clásico vs. transformer con tiempos. | `detectar_objetos`, `segmentar_semantica`, `segmentar_con_sam`, `comparar_metodos` |
| `pipeline` | Encadena las 7 etapas y devuelve el resultado de cada una. | `ejecutar_pipeline` |

---

## Modelos transformer

| Uso | Modelo | Parámetros | Enlace |
|---|---|---|---|
| Detección (por defecto) | RT-DETR R50-VD (COCO) | 42.9 M | [PekingU/rtdetr_r50vd](https://huggingface.co/PekingU/rtdetr_r50vd) |
| Detección (opcional) | DETR ResNet-50 (COCO) | 41.5 M | [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) |
| Segmentación semántica | SegFormer-B0 (ADE20K) | 3.8 M | [nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) |
| Segmentación por prompt (por defecto) | SlimSAM 50 % uniforme | 28.0 M | [nielsr/slimsam-50-uniform](https://huggingface.co/nielsr/slimsam-50-uniform) |
| Segmentación por prompt (opcional) | SAM ViT-B | 93.7 M | [facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base) |

Tiempos de referencia de la red sola, después del calentamiento, en una imagen de 960 px (Apple
Silicon con MPS): RT-DETR ≈ 90 ms, DETR ≈ 150–260 ms, SegFormer-B0 ≈ 15–55 ms, SlimSAM ≈ 0.5 s,
SAM ViT-B ≈ 0.7 s. En CPU son entre 2 y 10 veces mayores. La pestaña **8 · Transformers** mide y
compara los tiempos en tu máquina.

---

## Pruebas

```bash
pytest                 # 100 pruebas rápidas (sin descargar modelos)
pytest -m slow         # 3 pruebas que descargan y ejecutan los modelos de Hugging Face
```

Qué se verifica, entre otras cosas:

- La calibración recupera la K y la distorsión con las que se generaron las vistas sintéticas (error < 3 %).
- La proyección pinhole y la DLT manuales coinciden con `cv2.projectPoints` y `cv2.getPerspectiveTransform`.
- La convolución manual coincide con `cv2.filter2D` (diferencia < 1e-9).
- La ecualización manual y Otsu manual coinciden con `cv2.equalizeHist` y `THRESH_OTSU` (±1 nivel).
- Erosión y dilatación manuales coinciden con OpenCV; dualidad e idempotencia de la apertura.
- Watershed separa objetos que se tocan; las excentricidades coinciden con scikit-image; Hu es
  invariante a rotación y escala.
- La homografía de la media cancha ubica al equipo propio con error < 10 cm y al rival con error < 40 cm.
- Con la homografía, la selección del equipo propio coincide exactamente con la verdad de la escena
  sintética; sin ella, elige el grupo de camisetas más cercano a la cámara.
- El pipeline clásico se queda solo con el equipo propio (todos en la media cancha propia), descarta al
  rival y encuentra el balón.
- La app de Gradio se construye y cada callback de la UI se ejecuta sin excepciones con la escena
  sintética (sin levantar el servidor); el modo guiado da un resumen con los jugadores de tu equipo,
  el balón y su zona (ataque o defensa), y sin esquinas no dibuja el mapa.

---

## Datos de ejemplo

- `data/samples/cancha_sintetica*.jpg` y `rally_sintetico.mp4`: escenas sintéticas tomadas desde detrás
  de la línea de fondo propia, con posiciones conocidas (cancha naranja, zona libre verde, equipo propio
  rojo de espaldas con su número, rival azul, balón amarillo con franjas azules).
- `data/samples/tablero/`: 14 fotos sintéticas de un tablero 9 × 6 tomadas con una cámara de K y
  distorsión conocidas. `tablero_plano.png` es el patrón para imprimir.
- `data/samples/descargadas/` (no versionada): fotos reales de Wikimedia Commons con licencia libre
  (CC0, dominio público, CC BY, CC BY-SA). El script escribe `CREDITOS.md` con autor, licencia y enlace
  de cada una. El filtro es por título, así que conviene revisar las fotos y borrar las que no sean de
  un partido.

---

## Limitaciones

- **Cambio de dominio**: los jugadores sintéticos son figuras geométricas y RT-DETR/DETR detectan pocos.
  En fotos reales ocurre lo contrario: los transformers detectan bien y el método clásico confunde al
  público con jugadores.
- **Segmentación clásica de jugadores**: supone que los colores dominantes son el fondo (piso, zona
  libre, gradas). Falla con público muy colorido, jugadores superpuestos o colores parecidos al piso.
- **Solo el equipo propio**: el rival no se analiza; desde atrás queda lejos y tapado por la red y por la
  primera línea propia. Con las esquinas, son propios los que pisan la media cancha propia (con 0,5 m de
  margen en la red); sin ellas, el grupo de camisetas con los pies más abajo en la imagen.
- **Oclusiones de la cámara trasera**: el método clásico puede unir a dos jugadores propios que se tapan
  (encuentra 5–6 de los 6 propios en las escenas sintéticas). Una cámara elevada (tribuna o trípode
  alto) reduce el problema.
- **Balón**: los rangos HSV están pensados para balones amarillo/azul; otros colores requieren ajustar
  los sliders.
- **Homografía**: supone el piso plano y las 4 esquinas de la media cancha propia visibles (las dos de la
  línea central bajo la red y las dos de la línea de fondo propia). Solo los puntos del piso se proyectan
  bien: un balón en el aire no se puede ubicar en el mapa cenital con una sola cámara.
- **Calibración**: el ejemplo usa un tablero sintético. Para una cámara real hay que fotografiar un
  tablero impreso con esa misma cámara y resolución.
- **Sin esquinas**: el equipo propio se reconoce por el color de la camiseta, así que se asume que los dos
  uniformes son de colores distintos; el líbero y los árbitros pueden quedar mal asignados.
- **Video**: se procesa frame por frame, sin seguimiento (*tracking*) entre frames.
- **Recursos**: la primera ejecución descarga unos 850 MB de pesos si se usan todos los modelos; SAM
  ViT-B es lento en CPU (por eso SlimSAM va por defecto).

---

## Licencia

[MIT](LICENSE) © 2026 Jorge Enrique Olaya Liévano. Los modelos de Hugging Face conservan sus licencias
originales (ver la ficha de cada modelo) y las fotos descargadas, las suyas (ver `CREDITOS.md`).
