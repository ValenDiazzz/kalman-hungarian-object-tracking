# Animación educativa: Kalman + Hungarian

Archivo: `outputs/figures/kalman_hungarian_explained.gif`, ignorado por Git.
Formato vertical 900 × 1200 píxeles, 68 etapas visuales, duración 45.05 s y bucle.
Módulo: [`explained_animation.py`](../src/kalman_hungarian_object_tracking/explained_animation.py).

## Versión compacta v2

Archivo: `outputs/figures/kalman_hungarian_explained_v2.gif`, **16.6 s**, 13
imágenes: pregunta inicial y cuatro etapas para los frames **56, 60 y 64** de
`crossing/100`. Ambas versiones usan formato 900 × 1200, colores, símbolos
y la misma fórmula de las covarianzas.

```bash
.venv/bin/python -m kalman_hungarian_object_tracking.explained_animation_v2 \
  --benchmark-dir outputs/benchmark \
  --output outputs/figures/kalman_hungarian_explained_v2.gif
```

El [módulo](../src/kalman_hungarian_object_tracking/explained_animation_v2.py)
reutiliza el replay verificado de la primera versión. Procesa los 120 frames,
las 706 observaciones y todos los tracks; compara 706 asociaciones y 855 estados
con el archivo guardado antes de dibujar. El replay procesa también los frames
entre los tres instantes mostrados.

Los límites de ambos ejes son **[45, 55] u** en todas las etapas. Observación
solo muestra realidad sintética y detecciones; predicción muestra medias y
elipses previas; asociación añade detecciones y líneas efectivamente elegidas;
actualización muestra medias y elipses posteriores, con predicciones atenuadas
para apreciar la corrección. Las tarjetas distinguen las etapas mediante título
y fondo. La pregunta inicial es «¿Cómo sabemos qué detección corresponde a
cada objeto?».

Las etiquetas ID 4 e ID 2 acompañan las medias predichas o actualizadas mediante
guías finas punteadas. Se ubican fuera de las elipses y evitan otros rótulos y
marcadores. Las líneas gruesas continuas representan exclusivamente asociaciones.
En observación no se ponen IDs a las detecciones aún no asociadas.

Hungarian minimiza el objetivo de la asignación aumentada de **todos** los tracks
y detecciones: suma de costos Mahalanobis² de pares admitidos, más penalidades
por tracks/detecciones sin asignar. En `augmented_cost_matrix` la penalidad es
`lambda=gate_cost+1`; las aristas fuera de la puerta están prohibidas. No optimiza
solo las dos parejas dibujadas ni garantiza identidades verdaderas correctas.

Se revisaron imágenes decodificadas del GIF a **360 × 480** al principio, en el
cruce y al final. Se ajustó un rótulo próximo a una detección durante asociación.
El manifiesto, tiempos decodificados, hashes y previews se guardan bajo
`outputs/explained-review-v2/`, separado de los resultados del benchmark.

```bash
.venv/bin/python -m pytest -q tests/test_explained_animation.py \
  tests/test_explained_animation_v2.py tests/test_kalman.py \
  tests/test_assignment.py tests/test_tracker.py
```

Resultado ejecutado: **54 tests pasaron**. Las verificaciones del render v2
consumen el ensayo existente; si falta su archivo local, se omiten sin generar
datos. Validan coordenadas de las líneas, anclajes y separación de rótulos,
ejes constantes, duración codificada y conservación de la animación original.

Texto alternativo: tres instantes de dos objetos que se cruzan. Las detecciones
ruidosas aparecen primero; luego las predicciones azules y naranjas con su
incertidumbre. Hungarian conecta las parejas seleccionadas y Kalman actualiza
las medias. Los IDs acompañan a los tracks, que terminan en lados opuestos.
El extracto conserva las limitaciones de interpretación descritas abajo.

## Reproducción de la versión original

Desde la raíz, con el entorno instalado y los artefactos del benchmark existentes:

```bash
.venv/bin/python -m kalman_hungarian_object_tracking.explained_animation \
  --benchmark-dir outputs/benchmark \
  --seed 100 \
  --output outputs/figures/kalman_hungarian_explained.gif
```

No genera datos ni ejecuta el benchmark. El módulo compara los hashes de los
algoritmos originales, configuración y CSV antes de reconstruir. Reproduce
`kf_hungarian` sobre las 706 detecciones de los 120 frames de `crossing/100`,
incluidos los cuatro objetos alejados y todos los tracks activos. Comprueba
todas las asociaciones, nacimientos y estados contra los registros guardados.
Las etiquetas GT se usan después para seleccionar el detalle dibujado y mostrar
las posiciones verdaderas; no entran a `Tracker.step` ni al solver.

## Qué muestra cada elemento

| Elemento | Significado |
|---|---|
| `+` gris | Posición verdadera sintética, como referencia visual |
| `×` oscuro | Detección ruidosa original del CSV |
| Rombo hueco azul o naranja | Media predicha de Kalman antes de incorporar la detección |
| Línea de color | Asociación real elegida por el solver, entre predicción y detección |
| Círculo lleno azul o naranja | Media del estado después de la actualización |
| Contorno discontinuo | Incertidumbre de posición antes de actualizar |
| Contorno continuo | Incertidumbre de posición después de actualizar |

Azul identifica T4 y naranja T2 durante el extracto. Los colores representan
tracks; no redefinen identidades verdaderas. Las etiquetas de ID permanecen en
posiciones fijas fuera de los puntos para evitar superposición. Las flechas
discontinuas de la etapa de predicción unen la estimación previa con la media
predicha; no son asignaciones a detecciones.

Cada frame del extracto se presenta en cuatro pausas: **observación, predicción,
asociación y actualización**. El primer ciclo se mantiene más tiempo para
aprender los símbolos; frames 58–62 tienen pausas más largas y zoom gradual.
La vista cambia de escala, conservando unidades y aspecto geométrico 1:1.
Dentro de un mismo frame, las cuatro etapas comparten exactamente el viewport.
La predicción y la actualización se muestran como etapas discretas, sin interpolación.
La duración artística del GIF no representa el tiempo físico ni la latencia.

## Covarianzas y límites de interpretación

El contorno utiliza la covarianza de **posición del estado** `C=H P Hᵀ`,
antes o después del update según la etapa. Con `q=chi2.ppf(0.95,2)`, sus
semiejes son `sqrt(q*lambda_1)` y `sqrt(q*lambda_2)`; la orientación viene
del autovector del mayor autovalor. Ambos ejes espaciales tienen la misma escala.
Los contornos se calculan a tamaño real en unidades `u`. En este modelo
isotrópico se ven como circunferencias: son el caso de una elipse con semiejes
iguales.

Son regiones del 95% **bajo el modelo gaussiano**, no una cobertura empírica
comprobada. No son la puerta de asociación: esa usa
`S=H P⁻ Hᵀ+R` y el umbral Mahalanobis² 9.21034. La covarianza dibujada no
añade R; esa distinción se verifica también con un caso analítico en tests.

La semilla 100 conserva el ejemplo preseleccionado del protocolo. Se muestran
los frames 52–68, alrededor del cruce en 60, para concentrar la narrativa.
Es un extracto educativo, no una muestra representativa de rendimiento: fuera
de este intervalo existen nacimientos y alternancias con T8 en frames 47–51.
Esos tracks siguen participando en el replay completo aunque no se destaquen.
El solver decide sobre **todas** las detecciones y tracks; no sobre los dos
objetos que se dibujan. No se presenta el ejemplo como prueba de superioridad.

## Verificación e inspección

El generador guarda `outputs/explained-review/manifest.json` con hashes,
cantidad de observaciones/estados contrastados, intervalos, IDs mostrados,
cuantil de las elipses y tiempos de las etapas. También decodifica el GIF y
extrae sus cuatro etapas para frames 52, 60 y 68, tanto a resolución original
como a tamaño móvil 360 × 480, bajo `outputs/explained-review/previews/`.

La inspección móvil detectó inicialmente falta de separación de símbolos en
el cruce. Se agregó zoom y mayor énfasis de las líneas; durante asociación
las elipses y la verdad se atenúan, conservando su geometría. Se revisaron
los fotogramas del principio, cruce y final de la versión resultante.
El benchmark, los CSV y las figuras anteriores conservan sus hashes.

Pruebas matemáticas y de reproducción de la animación:

```bash
.venv/bin/python -m pytest -q tests/test_explained_animation.py
```

Las pruebas crean un archivo temporal de cuatro frames en una ruta ignorada;
no dependen de versionar un dataset ni de volver a ejecutar el benchmark.
Validan la elipse con una covarianza rotada, su frontera por forma cuadrática,
rechazo de covarianzas inválidas, exclusión de etiquetas del tracker,
procesamiento de todas las observaciones, integridad de los CSV y las etapas GIF.

Referencias de implementación gráfica:
[Matplotlib, elipses de covarianza](https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html)
y [Pillow, GIF y duraciones por frame](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif).
