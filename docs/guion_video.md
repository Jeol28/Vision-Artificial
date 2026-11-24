# Guion del video – VolleyVision (8 a 10 minutos)

**Antes de grabar**

- `source .venv/bin/activate && streamlit run volleyvision/ui/app.py`
- Abrir una vez las páginas 6 (SegFormer y SlimSAM) y 8 (RT-DETR y DETR) para que los modelos ya estén
  descargados y en caché: así los tiempos que se ven en el video son los de inferencia.
- Tener abiertos en el editor `README.md` y la carpeta `volleyvision/`, y una terminal con
  `git log --oneline --graph --all` listo.
- Imagen de ejemplo principal: `cancha_sintetica.jpg`. Foto real: `voleibol_03.jpg` (o la que mejor
  se vea de `data/samples/descargadas/`).

---

## 0:00 – 0:45 · Introducción

**En pantalla:** página **Inicio** de la app.

> "Hola, soy Jorge Enrique Olaya Liévano y este es mi proyecto final de Visión Artificial:
> **VolleyVision**. Es una aplicación que analiza fotos y video de voleibol con un pipeline de visión
> por computador, con la cámara detrás de la línea de fondo de mi equipo: se nos ve de espaldas, con
> el número de la camiseta, y al rival de frente. Cada tema del curso es una etapa: calibración,
> filtros, histogramas, umbralización,
> morfología, segmentación y caracterización. Al final comparo las técnicas clásicas con modelos
> transformer de Hugging Face, que uso solo en inferencia con pesos públicos, sin entrenar nada."

Señalar la tabla del recorrido y el diagrama del pipeline.

## 0:45 – 1:45 · Arquitectura y ramificación

**En pantalla:** `README.md` (secciones *Arquitectura* y *Estrategia de ramificación*) y la terminal.

> "El código está en tres capas. En `core` cada tema es un módulo con funciones puras: reciben una
> imagen y devuelven imágenes o datos, sin Streamlit. La UI solo llama al core, y `utils` tiene la
> carga de imágenes, el dibujo y un generador de escenas sintéticas con verdad conocida, que uso en
> las pruebas."

- Mostrar el árbol de `volleyvision/` (core, ui, utils).
- Ejecutar `git log --oneline --graph --all`:
  > "Trabajé con `main` estable, `develop` de integración y una rama `feature` por módulo, con
  > Conventional Commits y merges `--no-ff`, por eso cada rama queda visible en el historial. La
  > versión entregada tiene la etiqueta v1.0.0."
- Ejecutar `pytest -q` y mostrar que pasan las 76 pruebas.

## 1:45 – 2:45 · Calibración

**Código:** `volleyvision/core/calibracion.py` → docstring del módulo (modelo pinhole),
`calibrar_camara` y `homografia_cancha`.

> "El modelo pinhole proyecta un punto 3D con K por [R|t]. Con varias fotos de un tablero, el método
> de Zhang despeja K y luego minimiza el error de reproyección."

**UI:** página **1 · Calibración**.

1. Pulsar *Calibrar cámara*. Mostrar K, los coeficientes de distorsión y el error RMS (< 0.5 px).
   Comparar con la K verdadera que aparece abajo.
2. Mover el slider α y mostrar cómo se enderezan las rectas del tablero.
3. Pestaña *Homografía*: hacer clic en las 4 esquinas de la media cancha propia en el orden indicado
   (las dos de la red: sup-izq y sup-der; las dos de mi línea de fondo: inf-der e inf-izq), mostrar el
   botón *Reiniciar puntos* y la vista cenital.
   > "Desde atrás veo las 4 esquinas de mi media cancha. Como el piso es un plano, eso basta para una
   > homografía, y esa homografía sirve para todo el piso, incluso el campo rival. La resuelvo también
   > con mi propia DLT y coincide con OpenCV."

## 2:45 – 3:30 · Filtros

**Código:** `core/filtros.py` → `convolucion_manual` (las tres líneas: `np.pad`, `sliding_window_view`, `einsum`).

**UI:** página **2 · Filtros**.

1. *Suavizado*: agregar ruido sal y pimienta; comparar mediana con bilateral.
2. *Convolución manual*: elegir "Sobel X", editar un valor del kernel y mostrar que la diferencia con
   `cv2.filter2D` es ~1e-13 y los tiempos de ambas.
3. *Líneas de la cancha*: bilateral → Canny → Hough.

## 3:30 – 4:10 · Histogramas

**Código:** `core/histogramas.py` → `ecualizar_manual` y `asignar_equipos`.

**UI:** página **3 · Histogramas**.

1. *Ecualización y CLAHE*: comparar las tres imágenes y la métrica "desviación del brillo entre bloques".
   > "CLAHE corrige la iluminación desigual del coliseo sin amplificar el ruido."
2. *Uniformes*: mostrar la matriz de Bhattacharyya y los equipos A/B.

## 4:10 – 4:50 · Umbralización

**Código:** `core/umbralizacion.py` → `otsu_manual` (sumas acumuladas y la fórmula de σ²_B).

**UI:** página **4 · Umbralización**.

1. *Otsu*: curva σ²_B(t) con el T manual igual al de OpenCV.
2. *Adaptativo* contra el global.
3. *HSV*: preset "Balón (amarillo)" y mover los sliders de H.

## 4:50 – 5:30 · Morfología

**Código:** `core/morfologia.py` → `erosion_manual` (mínimo del vecindario) y `limpiar_mascara`.

**UI:** página **5 · Morfología**.

1. Máscara "Jugadores" con 1–2 % de ruido: aplicar *Apertura* y mostrar cómo baja el número de
   componentes.
2. Cambiar la forma del elemento estructurante (elipse, cruz, rectángulo) y el tamaño.
3. Mostrar la cuadrícula con todas las operaciones y la línea "Erosión manual = cv2.erode: sí".

## 5:30 – 6:40 · Segmentación

**Código:** `core/segmentacion.py` → `watershed_distancia` y `segmentar_jugadores_clasico`;
`core/transformers_models.py` → `segmentar_con_sam`.

**UI:** página **6 · Segmentación**.

1. *K-means*: k = 5 en Lab.
2. *Watershed*: demo de los dos balones que se tocan: contornos encuentra 1 objeto, watershed 2.
3. *Jugadores (clásico)*: verde = acierto y rojo = error, usando la verdad de la escena sintética.
4. *SegFormer*: activar y mostrar la clase "person" y el tiempo.
5. *SAM*: hacer clic sobre el balón y luego sobre un jugador, y mostrar las 3 máscaras con su IoU estimado.

## 6:40 – 7:20 · Caracterización

**Código:** `core/caracterizacion.py` → `describir_region` (circularidad, excentricidad con valores
propios, solidez, Hu) y `es_balon`.

**UI:** página **7 · Caracterización**.

1. *Descriptores*: seleccionar una región y mostrar centroide, caja y envolvente convexa.
2. *¿Es balón?*: el balón sale en verde y las camisetas azules en rojo con su motivo; bajar la
   circularidad mínima para mostrar cómo cambian los aciertos.
3. *Proyección a la cancha*: mapa cenital con mi equipo abajo y el rival arriba.
   > "Uso el punto de los pies porque la homografía solo vale en el piso. Desde atrás, mi primera línea
   > tapa a los rivales de la red: el método clásico une esas siluetas, y es una limitación de esta toma."

## 7:20 – 8:30 · Transformers: clásico vs. transformer

**Código:** `core/transformers_models.py` → `detectar_objetos` y `comparar_metodos`;
`ui/modelos.py` → `st.cache_resource`.

**UI:** página **8 · Transformers**.

1. *Detección* con `voleibol_03.jpg`: RT-DETR detecta personas y el balón; cambiar a DETR.
2. *Comparación* con la foto real: pulsar *Comparar* y mostrar las tres columnas y la gráfica de tiempos.
   > "En la foto real el método clásico confunde al público con jugadores y no encuentra el balón; los
   > transformers sí, y RT-DETR es más rápido que DETR."
3. *Comparación* con `cancha_sintetica.jpg`:
   > "Aquí pasa lo contrario: los jugadores sintéticos son figuras geométricas y los transformers
   > detectan pocos. Es un cambio de dominio, una de las limitaciones que documento en el README."

## 8:30 – 9:30 · Pipeline completo

**Código:** `core/pipeline.py` → `ejecutar_pipeline` (las 7 etapas y la dataclass `Paso`).

**UI:** página **9 · Pipeline**.

1. Fuente *Video (elegir frame)* → `rally_sintetico.mp4`, mover el slider de frame.
2. Dejar **desactivada** *Corregir distorsión con la calibración* y explicar por qué:
   > "Esta corrección solo se aplica a imágenes tomadas con la cámara que calibré; la escena sintética
   > no tiene distorsión, así que corregirla la deformaría."

   Usar las esquinas de ejemplo, activar RT-DETR y ejecutar.
3. Recorrer los pasos 1 a 7 y la gráfica de tiempo por etapa, y terminar en el mapa cenital.

## 9:30 – 10:00 · Cierre

> "En resumen, VolleyVision implementa cada tema del curso como una etapa de un pipeline real, con
> pruebas que comparan mis implementaciones manuales con OpenCV, y muestra con datos cuándo conviene
> una técnica clásica y cuándo un transformer. El código, el README y este guion están en el
> repositorio. Gracias."
