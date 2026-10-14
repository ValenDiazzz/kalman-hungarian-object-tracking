# Kalman Hungarian Object Tracking

Seguimiento multiobjeto online en una cinta transportadora **simulada**:
seis objetos 2D, un detector idealizado con ruido, omisiones y falsos positivos,
y cuatro métodos que asocian detecciones a identidades persistentes. Es un
experimento educativo de CPU, sin cámara, detector visual ni validación
industrial. La pregunta es cuándo ayudan la predicción y la asignación global
a conservar identidades.

El benchmark incluye 4 escenarios,
20 semillas finales y 4 métodos (320 corridas), además de 48 corridas de
desarrollo. No hubo ajuste de parámetros ni selección retrospectiva de semillas.
Todos los datos y artefactos generados permanecen en `outputs/` ignorado.

## Instalación y ejecución

Python >=3.11 y Git. Entorno comprobado: Python 3.12.3.
Desde la raíz del repositorio:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m mot_demo.cli simulate --scenario crossing --seed 100 --output-dir outputs/data
python -m mot_demo.cli benchmark --config configs/benchmark.json --output-dir outputs/benchmark
python -m mot_demo.cli visualize --benchmark-dir outputs/benchmark --output-dir outputs/figures
python -m mot_demo.cli animate --scenario crossing --seed 100 --method kf_hungarian --benchmark-dir outputs/benchmark --output outputs/figures/crossing_kf_hungarian.gif
```

La implementación vive en `kalman_hungarian_object_tracking`;
`mot_demo.cli` es una capa de compatibilidad sin lógica científica duplicada.
También están disponibles `mot-demo` y
`python -m kalman_hungarian_object_tracking.cli`. Cada subcomando tiene
`--help`. La animación usa PillowWriter, sin FFmpeg.

Para fijar las versiones del entorno ejecutado, instalar primero
`requirements-reproducible.txt` y luego el editable con `--no-deps`.
Los rangos compatibles están en `pyproject.toml`; las versiones científicas
exactas y la plataforma se registran en metadata. Los pines corresponden al
entorno validado con Python 3.12, no aseguran instalación en toda plataforma.

## Los cuatro métodos

| Método | Posición de referencia | Costo y puerta | Asociación |
|---|---|---|---|
| nn_greedy | Última detección observada | Euclídea² <= 25 u² | Greedy |
| nn_hungarian | Última detección observada | Euclídea² <= 25 u² | LSAP aumentado |
| kf_greedy | Predicción Kalman | Mahalanobis² <= 9.21034 | Greedy |
| kf_hungarian | Predicción Kalman | Mahalanobis² <= 9.21034 | LSAP aumentado |

Comparar greedy y Hungarian **dentro de cada familia** aísla la estrategia
de asignación. Comparar NN y KF cambia simultáneamente predictor, costo
y puerta; no permite atribuir el resultado únicamente a Kalman.

NN conserva la última observación durante misses, sin extrapolar. KF predice
también en los frames sin detección. Cada detección no asociada crea
inmediatamente un track; no hay confirmación ni apariencia. Los IDs crecen sin
reciclarse. Un match reinicia `missed`; se permiten ocho ausencias consecutivas
y se elimina al noveno miss (`missed > max_missed`). Los estados activos se
guardan aun sin observación; las asociaciones solo existen para detecciones.

## Modelo y asociación

Estado `s=[x,y,vx,vy]ᵀ`, posiciones en unidades arbitrarias u, velocidades
en u/frame y `dt=1`:

```text
F = [[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]]
H = [[1,0,0,0], [0,1,0,0]]
G = [[dt²/2,0], [0,dt²/2], [dt,0], [0,dt]]
Q = sigma_a² G Gᵀ ; R = sigma_det² I₂
s⁻ = F s ; P⁻ = F P Fᵀ + Q
r = z - H s⁻ ; S = H P⁻ Hᵀ + R
K = P⁻ Hᵀ S⁻¹ ; s = s⁻ + K r
P = (I-KH) P⁻ (I-KH)ᵀ + K R Kᵀ
```

El código resuelve sistemas lineales con `solve`, sin inversión explícita;
usa Joseph, simetriza P y comprueba finitud y semidefinición positiva dentro
de tolerancia numérica. Nacimientos: `s=[zx,zy,0,0]`, `P0=diag(4,4,4,4)`.
`sigma_det=1 u`, `sigma_a=0.05 u/frame²`: supuestos configurados, sin
estimación sobre datos reales. La verdad no tiene aceleración; Q modela
incertidumbre. La puerta KF aproxima el cuantil χ² de 0.99 con dos grados
de libertad bajo hipótesis gaussianas, sin garantía con outliers.

Greedy ordena **todos** los pares admisibles por
`(costo,track_id,detection_id)` y acepta extremos libres. LSAP usa
`scipy.optimize.linear_sum_assignment`, un solver de la familia
Jonker–Volgenant, denominado Hungarian en los IDs de métodos.
Las aristas inadmisibles se bloquean **antes** de resolver. Con n tracks
y m detecciones, se minimiza sobre la matriz (n+m) × (n+m):

```text
                         detecciones m            misses n
tracks n                 C válido / BIG          diag(lambda) / BIG
nacimientos m            diag(lambda) / BIG       ceros m × n
```

`lambda=gate_cost+1`, BIG finito superior a cualquier solución válida
(1e9 en estas instancias). Fuera de diagonal, los bloques de penalización
son BIG. Solo se decodifican pares válidos del bloque superior izquierdo.
Sin tracks/detecciones, se procesan nacimientos/misses directamente.
El mínimo de este objetivo por frame **no** garantiza mínima pérdida de
identidad temporal ni equivale a maximizar la cardinalidad.

## Simulador y protocolo

`configs/benchmark.json`, experimento 1.0, contiene todas las trayectorias,
parámetros, semillas y rutas. Son 120 frames, seis objetos y región nominal
[0,100]² u, sin reflexión ni recorte. La verdad es determinista por escenario;
las semillas afectan únicamente al detector.

| Escenario | Geometría | Omisión aleatoria | Ausencia adicional | Falsos positivos |
|---|---|---:|---|---|
| separated | Seis trayectorias espaciadas | 2% | Ninguna | Ninguno |
| crossing | Dos objetos se cruzan en (50,50), frame 60; cuatro alejados | 2% | Ninguna | Ninguno |
| dropout | Misma geometría del cruce | 10% | GT 0, frames 57–62 inclusive | Ninguno |
| clutter | Misma geometría del cruce | 10% | Ninguna | Poisson(2)/frame, uniforme en [0,100]² |

El ruido es N(0,I₂) en todos los escenarios. No se hizo un barrido de
intensidad de ruido; las conclusiones se limitan a estos valores.
El generador usa `numpy.random.default_rng(seed)`, guarda primero la verdad
y baraja detecciones antes de asignar IDs. Los cuatro métodos reciben
**los mismos registros inmutables**, con solo frame, detection_id, x e y.
Las etiquetas verdaderas permanecen fuera del tracking y se unen después
para evaluar y visualizar.

Desarrollo: semillas [1,2,3], `--split dev`, directorio separado.
Evaluación: [100..119], sin ajuste posterior. El ejemplo visual crossing/100
se fijó antes de la evaluación.

## Métricas y límites

El score principal **detected_id_f1 no es IDF1 MOTChallenge** y no se calcula
MOTA ni CLEAR MOT. Evalúa asociación condicionada a detecciones presentes,
excluyendo objetos de los frames en que el detector falló.

Se construye M[g,p], conteo de detecciones verdaderas de g asignadas a p.
Un matching global de identidades uno-a-uno maximiza la suma de conteos,
con vacíos para tamaños desiguales. Se usa **solo para puntuar**:
`IDTP=sum M[g,p]`, `IDFN=N_gt_det-IDTP`, `IDFP=N_pred_det-IDTP`.
`N_pred_det` incluye asociaciones a falsos positivos.

- `detected_id_f1=2*IDTP/(N_gt_det+N_pred_det)`; NaN si ambos son cero.
- `id_switches`: cambio respecto del track de la detección anterior observada
  del mismo GT, incluso si hubo frames omitidos. No cuenta el primer avistamiento.
  `id_switches_per_100_detections=100*switches/N_gt_det`; NaN si no hay verdad observada.
- `track_purity=sum_p max_g M[g,p]/N_gt_det`; NaN si no hay verdad observada.
  Es diagnóstica: puede ser alta pese a fragmentación o alternancia entre
  varios tracks puros del mismo objeto. Los FP se informan por separado.
- `tracks_created`, `false_positive_detections`, `n_true_detections` y
  `detection_recall=N_gt_det/(n_frames*n_objects)`.
- `tracker_time_ms`: procesamiento de frames, sin simulación, scoring, IO
  ni visualización; depende de máquina, carga, librerías y orden fijo.

## Resultados observados

320 corridas test completas; 20 semillas por celda. Medianas de
detected_id_f1, derivadas de `outputs/benchmark/summary.csv`:

| Escenario | nn_greedy | nn_hungarian | kf_greedy | kf_hungarian |
|---|---:|---:|---:|---:|
| separated | 0.977377 | 0.977377 | 0.918738 | 0.918738 |
| crossing | 0.857454 | 0.838510 | 0.890205 | 0.910566 |
| dropout | 0.840785 | 0.840785 | 0.893420 | 0.893420 |
| clutter | 0.619192 | 0.639139 | 0.641957 | 0.659993 |

Comparación principal KF: Hungarian − greedy en medianas de F1 es +0.020361
en crossing, +0.018036 en clutter y cero en separated/dropout. Son diferencias
de medianas, **no** medianas de diferencias pareadas ni evidencia de
significancia estadística. NN-Hungarian empeora la mediana frente a NN-greedy
en crossing (−0.018945), aunque mejora en clutter (+0.019947).

Las métricas no coinciden en un ranking universal. En crossing, switches por
100 observaciones verdaderas (mediana) son 1.281466, 1.199080, 8.226363 y
7.943077, en el orden de métodos anterior. KF mejora F1 respecto de NN en
ese escenario y tiene más switches. En separated, NN obtiene mayor F1.
El comportamiento está condicionado por puertas y nacimientos inmediatos:
una observación rechazada puede crear un track duplicado que compite con
otro track del mismo objeto. La alta pureza no resuelve esa fragmentación.

El ejemplo crossing/100 produjo F1 de 0.917847, 0.923513, 0.916431 y 0.924929,
con 27, 23, 61 y 57 switches respectivamente. Es un ejemplo preseleccionado,
no la mejor corrida. Los gráficos incluyen todas las semillas, sin exclusiones.
Distribuciones, pureza, tracks, latencias y características del detector están
en [RESULTS.md](docs/RESULTS.md), trials y summary. No hay garantía de
superioridad universal, significancia ni transferencia a una instalación real.

## Artefactos y reproducción completa

Todos permanecen ignorados:

- `outputs/data/`: ground_truth y detections por escenario/semilla.
- `outputs/benchmark/`: assignments, track_states, trials, summary, protocol,
  config_snapshot y metadata con SHA-256.
- `outputs/figures/crossing_tracks.png`: cuatro métodos, mismo crossing/100,
  con detalle del cruce y trayectorias verdaderas.
- `outputs/figures/id_f1_by_scenario.png`: boxplots y veinte puntos por método.
- `outputs/figures/id_switches_by_scenario.png`: distribución de switches/100.
- `outputs/figures/crossing_kf_hungarian.gif`: 120 frames, 10 fps; detecciones,
  estados observados/predichos, IDs y verdad sintética.
- `outputs/benchmark/post_evidence.md` y `publication/`: evidence_summary,
  verified_metrics, figure_manifest y limitations generados desde resultados.

La animación educativa `outputs/figures/kalman_hungarian_explained.gif` presenta
las cuatro etapas de Kalman + Hungarian en formato vertical, con dos objetos
destacados, covarianzas de posición y asociaciones verificadas de crossing/100.
Su comando de reproducción y símbolos están en
[EXPLAINED_ANIMATION.md](docs/EXPLAINED_ANIMATION.md).
La versión compacta `outputs/figures/kalman_hungarian_explained_v2.gif` dura
16.6 segundos, conserva ejes fijos y muestra tres instantes del cruce con IDs
que acompañan a los tracks. Se regenera con el módulo `explained_animation_v2`;
el comando completo está en la misma documentación.

Para ejecutar desarrollo y comprobar la repetición:

```bash
python -m mot_demo.cli benchmark --config configs/benchmark.json --split dev --output-dir outputs/development
python -m mot_demo.cli benchmark --config configs/benchmark.json --output-dir outputs/benchmark-repeat
python scripts/verify_benchmark.py --benchmark-dir outputs/benchmark --compare-with outputs/benchmark-repeat --figures-dir outputs/figures --require-full
```

La verificación recalcula métricas y resúmenes, comprueba 320 trials y 16 filas
de summary, igualdad entre métodos del detector, hashes, PNG decodificables
y 120 frames GIF. Compara repetición excluyendo tiempo/fecha.
Las escrituras son atómicas por archivo; no correr dos benchmarks a la vez
sobre los mismos destinos. Ver [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
y [VALIDATION.md](docs/VALIDATION.md) para contratos, tolerancias y comandos.

## Arquitectura

```text
configs/benchmark.json
src/kalman_hungarian_object_tracking/
  config.py          validación del protocolo
  simulation.py      verdad y observaciones reproducibles
  association.py     NN/Mahalanobis, greedy, matriz aumentada y LSAP
  kalman.py          modelo, predict/update y covarianzas
  tracker.py         Detection, Association, TrackState y ciclo de vida
  metrics.py         evaluación aislada con etiquetas
  experiments.py     corridas, CSV, agregación, evidencia y metadata
  plots.py           PNG y GIF desde CSV guardados
  validation.py      aceptación e integridad de resultados
  io.py              rutas ignoradas y escritura atómica
  cli.py             argparse
src/mot_demo/cli.py   compatibilidad sin duplicación
tests/               pruebas científicas y controles de archivos
scripts/
  check_staged_paths.py
  run_gitleaks.py
  verify_benchmark.py
docs/                fuentes, reproducibilidad, validación, resultados, seguridad
outputs/             datos, figuras e informes locales ignorados
```

## Protección y licencia

`.gitignore` y el hook local restringen los archivos a las rutas del proyecto
y la documentación revisada. Para agregar documentación o una nueva carpeta
raíz, actualizar ambas listas. Los datasets nunca se versionan, tampoco los
sintéticos o fixtures de pruebas. Los tests generan archivos en rutas ignoradas.

Gitleaks **v8.30.0**: instalar desde la
[release oficial](https://github.com/gitleaks/gitleaks/releases/tag/v8.30.0),
verificar el SHA-256 contra sus checksums y colocar el ejecutable en .venv/bin/
o PATH. El wrapper prefiere la instalación local sin requerir activación.
Configuración de hooks por clon:

```bash
mkdir -p outputs/pre-commit-cache
export PRE_COMMIT_HOME="$PWD/outputs/pre-commit-cache"
.venv/bin/pre-commit validate-config
.venv/bin/pre-commit install
python scripts/check_staged_paths.py
```

El hook local rechaza rutas protegidas aunque se eluda .gitignore y comprueba
que los blobs staged .env.example contienen solo asignaciones ficticias:
CHANGE_ME, REPLACE_ME, EXAMPLE_ONLY o valor vacío; sin comentarios.
El hook oficial gitleaks-system está fijado a v8.30.0. Después de seleccionar
archivos públicos para staging:

```bash
python scripts/check_staged_paths.py
.venv/bin/pre-commit run
git status --short
```

Un índice vacío no certifica archivos nuevos ni todo el árbol. Herramientas
ausentes o fallos nunca equivalen a aprobación. Revisar contenido candidato,
archivos sin seguimiento e historial siguiendo [SECURITY.md](docs/SECURITY.md).

Código bajo [MIT](LICENSE).
[DEPENDENCIES.md](docs/DEPENDENCIES.md) registra licencias de dependencias.
[DATA_SOURCES.md](docs/DATA_SOURCES.md) documenta el generador y las referencias.
No hay canal privado de reporte establecido; sigue pendiente de definir por
el mantenedor antes de publicar.

La versión actual no cubre aceleraciones verdaderas, maniobras, oclusiones
prolongadas, apariencia, reidentificación, video real ni barridos de ruido.
Kalman puede fallar con modelos/covarianzas incorrectos; LSAP no optimiza el
futuro de las identidades. Ningún resultado constituye validación industrial.
