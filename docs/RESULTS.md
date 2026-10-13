# Resultados ejecutados: experimento 1.0

Fuente: `outputs/benchmark/trials.csv` y `summary.csv`, 320 corridas,
cuatro escenarios × veinte semillas [100..119] × cuatro métodos, 120 frames y
seis objetos por corrida. Desarrollo separado: 48 corridas, semillas [1,2,3].
Configuración original sin cambios de parámetros, exclusiones ni tuning.
SHA-256 de config: `c99c295a7d0565c3fcc02397173586d1fc7ab1fefac0eed90b49aea60e564e40`.

Entorno medido: Python 3.12.3, Linux 6.8.0-142-generic, x86_64, 24 CPUs lógicas.
NumPy 2.5.4, SciPy 1.18.1, pandas 3.0.6, Matplotlib 3.11.2, Pillow 12.3.0.
Fecha de la primera evaluación final: 2026-10-13 UTC. El código se ejecutó
con cambios locales sin commit; metadata registra la revisión base y hashes.
Tiempos totales por secuencia, sin IO ni evaluación, con orden fijo; no
comparables entre máquinas y sin un experimento de latencia dedicado.

## F1 y continuidad

`detected_id_f1` es un score propio condicionado a detecciones, no IDF1
MOTChallenge. Reportamos toda la distribución, no el mejor ejemplo.

| Escenario | Método | F1 mediana | F1 media | Desv. muestral | Mínimo | Máximo | Switches/100 mediana |
|---|---|---:|---:|---:|---:|---:|---:|
| clutter | kf_greedy | 0.641957 | 0.629539 | 0.050147 | 0.536042 | 0.719946 | 26.732077 |
| clutter | kf_hungarian | 0.659993 | 0.659519 | 0.054182 | 0.555701 | 0.755585 | 21.826678 |
| clutter | nn_greedy | 0.619192 | 0.616827 | 0.068227 | 0.498394 | 0.789542 | 5.369428 |
| clutter | nn_hungarian | 0.639139 | 0.631411 | 0.077769 | 0.496741 | 0.797851 | 5.417957 |
| crossing | kf_greedy | 0.890205 | 0.891087 | 0.067580 | 0.771509 | 0.991525 | 8.226363 |
| crossing | kf_hungarian | 0.910566 | 0.895663 | 0.068296 | 0.771509 | 1.000000 | 7.943077 |
| crossing | nn_greedy | 0.857454 | 0.880105 | 0.089360 | 0.728169 | 0.994342 | 1.281466 |
| crossing | nn_hungarian | 0.838510 | 0.871196 | 0.055780 | 0.781870 | 0.982979 | 1.199080 |
| dropout | kf_greedy | 0.893420 | 0.890022 | 0.069213 | 0.716511 | 0.979814 | 5.585890 |
| dropout | kf_hungarian | 0.893420 | 0.890258 | 0.068821 | 0.716511 | 0.979814 | 5.585890 |
| dropout | nn_greedy | 0.840785 | 0.859737 | 0.076055 | 0.742188 | 1.000000 | 0.780644 |
| dropout | nn_hungarian | 0.840785 | 0.858738 | 0.076113 | 0.742188 | 1.000000 | 0.703268 |
| separated | kf_greedy | 0.918738 | 0.901841 | 0.064839 | 0.791489 | 1.000000 | 7.357814 |
| separated | kf_hungarian | 0.918738 | 0.901841 | 0.064839 | 0.791489 | 1.000000 | 7.357814 |
| separated | nn_greedy | 0.977377 | 0.971628 | 0.027735 | 0.912181 | 1.000000 | 0.707550 |
| separated | nn_hungarian | 0.977377 | 0.971628 | 0.027735 | 0.912181 | 1.000000 | 0.707550 |

## Diagnósticos y cómputo

La pureza ignora FP en su numerador y puede esconder fragmentación.
Los tracks creados incluyen nacimientos causados por detecciones espurias.

| Escenario | Método | Pureza media | Tracks creados media | Tiempo mediana (ms/120 frames) |
|---|---|---:|---:|---:|
| clutter | kf_greedy | 0.991955 | 132.40 | 132.053 |
| clutter | kf_hungarian | 0.992498 | 127.75 | 133.380 |
| clutter | nn_greedy | 0.922795 | 217.20 | 7.792 |
| clutter | nn_hungarian | 0.938853 | 216.40 | 10.469 |
| crossing | kf_greedy | 0.991237 | 9.55 | 59.338 |
| crossing | kf_hungarian | 0.993708 | 9.50 | 62.285 |
| crossing | nn_greedy | 0.934812 | 7.55 | 4.541 |
| crossing | nn_hungarian | 0.914120 | 7.40 | 6.254 |
| dropout | kf_greedy | 0.992910 | 9.70 | 59.030 |
| dropout | kf_hungarian | 0.993141 | 9.70 | 61.212 |
| dropout | nn_greedy | 0.901695 | 7.20 | 4.529 |
| dropout | nn_hungarian | 0.889009 | 7.05 | 6.267 |
| separated | kf_greedy | 0.999364 | 9.55 | 50.532 |
| separated | kf_hungarian | 0.999364 | 9.55 | 53.402 |
| separated | nn_greedy | 1.000000 | 7.20 | 3.889 |
| separated | nn_hungarian | 1.000000 | 7.20 | 5.303 |

## Características del detector

Estas cantidades son idénticas entre variantes de cada escenario/semilla;
la tabla muestra medias sobre las veinte semillas, sin contar cuatro veces
cada simulación. La tasa de ruido sigma_det=1 es fija.

| Escenario | Detecciones verdaderas media | FP media | Recall de detección media |
|---|---:|---:|---:|
| clutter | 647.60 | 241.60 | 0.899444 |
| crossing | 707.15 | 0.00 | 0.982153 |
| dropout | 641.50 | 0.00 | 0.890972 |
| separated | 707.15 | 0.00 | 0.982153 |

## Comparaciones controladas e interpretación

Diferencias de medianas Hungarian − greedy (no medianas de diferencias pareadas):

| Escenario | Familia NN: ΔF1 | Familia KF: ΔF1 |
|---|---:|---:|
| separated | +0.000000 | +0.000000 |
| crossing | -0.018945 | +0.020361 |
| dropout | +0.000000 | +0.000000 |
| clutter | +0.019947 | +0.018036 |

La comparación principal KF favorece el F1 mediano de Hungarian en crossing y
clutter, y empata en los otros dos escenarios. No hay evidencia preespecificada
de significancia estadística. NN-Hungarian empeora el F1 mediano en crossing:
un mínimo del costo por frame no garantiza identidades futuras.

KF tiene más switches que NN en estos parámetros, incluso cuando obtiene mayor
F1. En separated la mediana NN es mayor. Una observación rechazada puede crear
un duplicado; ambos tracks coexisten y pueden alternarse sobre el mismo objeto.
En Kalman, una mayor covarianza también puede reducir el costo Mahalanobis de
una detección respecto de un track más preciso. Es una limitación del
objetivo y del ciclo de vida de los tracks. Las puertas son las del protocolo,
sin confirmación ni penalización extra por incertidumbre.

La pureza alta en KF no elimina fragmentación: múltiples tracks de un mismo
objeto pueden ser puros. F1 global, switches, pureza y tracks miden aspectos
distintos. KF vs NN cambia predictor, costo y puerta; no aísla el filtro.

## Ejemplo visual preseleccionado

crossing/seed=100 se mantuvo sin escoger otra corrida. En orden nn_greedy,
nn_hungarian, kf_greedy, kf_hungarian: F1 = 0.917847, 0.923513, 0.916431,
0.924929; switches = 27, 23, 61, 57. Las cifras de los paneles se leen del
trial correspondiente, y los puntos/IDs provienen de assignments.csv. El
GIF usa también track_states.csv, incluye los frames 0–119 y muestra ausencias
con estados predichos. Las distribuciones usan las veinte semillas completas.

Archivos locales:

- `outputs/figures/crossing_tracks.png`
- `outputs/figures/id_f1_by_scenario.png`
- `outputs/figures/id_switches_by_scenario.png`
- `outputs/figures/crossing_kf_hungarian.gif`
- `outputs/benchmark/post_evidence.md`
- `outputs/benchmark/publication/{evidence_summary.md,verified_metrics.json,figure_manifest.md,limitations.md}`

Todos los artefactos permanecen ignorados. Ninguna figura se promovió a una
ruta versionada. El resumen agregado de este documento se contrastó con CSV
ejecutados; no contiene filas de observaciones ni datasets.

## Límites y pendientes

Simulación pequeña de velocidad constante, no video ni mediciones industriales.
Sin detector real, apariencia, aceleraciones, reidentificación ni oclusiones
prolongadas evaluadas. No hay barrido de ruido ni evaluación fuera del modelo.
Los fallos del detector solo aparecen en recall/FP: F1 excluye frames no
observados. No se afirma superioridad universal, significancia ni robustez
industrial. Definir un canal privado de vulnerabilidades y decidir la
publicación corresponde al mantenedor. Ver REPRODUCIBILITY.md y VALIDATION.md
para comandos, contratos y comprobaciones.

## Validaciones realizadas

- `.venv/bin/python -m pytest -q`: 81 tests y 387 subcasos aprobados.
- `.venv/bin/ruff check src tests scripts/verify_benchmark.py`: aprobado.
- `.venv/bin/python -m pip check`: sin dependencias incompatibles.
- Verificación de desarrollo: 48 trials y 16 filas agregadas.
- Verificación final y repetición: 320 trials, 16 filas y 20 semillas por celda;
  métricas recalculadas, asociaciones/estados/datos y scores deterministas
  iguales; tiempo y fecha excluidos de la igualdad.
- Tres PNG decodificables y GIF de 120 frames decodificados con Pillow.
  Inspección visual de los tres PNG y del frame 60 del GIF; sin campos EXIF.
- Configuración y hooks de Git validados.
- Gitleaks 8.30.0: sin hallazgos en el árbol candidato completo y en el historial
  de todas las referencias locales. El índice quedó vacío; el hook staged
  por sí solo no certifica los archivos nuevos, cubiertos por el escaneo separado.

El escaneo automático complementa la revisión manual del contenido y no da
una garantía absoluta de seguridad.
