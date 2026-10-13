# Reproducibilidad

Python >=3.11; ejecución validada con Python 3.12.3 en CPU. Instalar en un
entorno virtual como indica el README. `pyproject.toml` declara rangos
compatibles; `requirements-reproducible.txt` fija las versiones del entorno
ejecutado. No se depende de APIs, modelos descargados ni FFmpeg.

## Protocolo congelado

La fuente de parámetros es `configs/benchmark.json`, experimento 1.0.
Trayectorias, puertas, covarianzas, semillas y tasas se fijaron antes de la
evaluación; no se ajustaron hiperparámetros. Desarrollo usa [1,2,3],
`--split dev` y un directorio separado. Evaluación final usa exclusivamente
[100..119]: 4 escenarios × 20 semillas × 4 métodos = 320 corridas de 120 frames.
El ejemplo gráfico es crossing/100, preseleccionado. Los tests de CLI usan
una configuración reducida creada en ejecución en outputs/test-tmp/, sin
alterar la configuración principal ni ajustar parámetros.

Cada trial genera una única secuencia de observaciones cuyos registros
inmutables se reutilizan entre métodos. Se registra el hash de la vista sin
etiquetas. Los IDs verdaderos se unen después del seguimiento, por frame y
detection_id. Los tiempos cubren solo `Tracker.step`, incluidos costos,
predicción, asignación, actualización y emisión; excluyen simulación, métricas,
CSV y gráficos. El orden de métodos es fijo. Las latencias son observaciones
del entorno, sin garantías de rendimiento ni comparaciones entre máquinas.

## Comandos

Desde la raíz, con .venv activado:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m mot_demo.cli benchmark --config configs/benchmark.json --split dev --output-dir outputs/development
python scripts/verify_benchmark.py --benchmark-dir outputs/development
python -m mot_demo.cli simulate --scenario crossing --seed 100 --output-dir outputs/data
python -m mot_demo.cli benchmark --config configs/benchmark.json --output-dir outputs/benchmark
python -m mot_demo.cli visualize --benchmark-dir outputs/benchmark --output-dir outputs/figures
python -m mot_demo.cli animate --scenario crossing --seed 100 --method kf_hungarian --benchmark-dir outputs/benchmark --output outputs/figures/crossing_kf_hungarian.gif
python -m mot_demo.cli benchmark --config configs/benchmark.json --output-dir outputs/benchmark-repeat
python scripts/verify_benchmark.py --benchmark-dir outputs/benchmark --compare-with outputs/benchmark-repeat --figures-dir outputs/figures --require-full
```

Para instalar las versiones registradas antes del editable:

```bash
python -m pip install -r requirements-reproducible.txt
python -m pip install --no-deps -e '.[dev]'
```

La CLI principal también admite
`python -m kalman_hungarian_object_tracking.cli` y `mot-demo`;
`mot_demo.cli` solo delega a la implementación. Los comandos requieren
la raíz del repositorio para resolver rutas relativas. Todos los destinos
generados deben permanecer bajo outputs/ del repositorio.

## Artefactos y comprobaciones

- outputs/data/: verdad y detecciones ordenadas por frame/detection_id.
- protocol.json: configuración, semillas y hashes del código; se escribe antes
  de evaluar la partición seleccionada.
- config_snapshot.json y metadata.json: parámetros, Python, versiones científicas,
  plataforma sin usuario/hostname, fecha UTC, revisión base y estado sin commit,
  hashes SHA-256 de datos, código y salidas.
- assignments.csv y track_states.csv: asociaciones por detección y estados
  activos, incluidos misses. Solo el primero incorpora etiquetas GT.
- trials.csv y summary.csv: métricas por corrida y media, mediana, desviación
  muestral (ddof=1), mínimo y máximo por escenario/método; n_seeds=20 en final.
- post_evidence.md y publication/: evidencia agregada automática, límites e
  índice de figuras para preparar material técnico.

La verificación recalcula scores desde asociaciones y compara el resumen con
trials. Verifica hashes, datos idénticos entre métodos, verdad contra el
generador, exclusión de GT del estado serializado y unicidad de asociaciones.
La repetición exige igualdad de datos, asociaciones, estados y métricas,
excluyendo tiempos y metadatos volátiles. Decodifica los tres PNG y todos los
frames del GIF. No valida significancia ni generalización industrial.

Las escrituras son atómicas por archivo, sin una transacción para todo el
directorio. No ejecutar benchmarks simultáneos sobre las mismas rutas. Ante
una interrupción, repetir el comando completo y verificar hashes. Una corrección
científica posterior exige repetir todas las semillas y variantes, sin selección
retrospectiva. Los informes privados quedan ignorados; seguridad de Git en
SECURITY.md.
