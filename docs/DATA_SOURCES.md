# Procedencia de datos

No se descargan datasets ni se usan mediciones, videos o datos industriales.
Todas las trayectorias y observaciones se generan en ejecución con
`kalman_hungarian_object_tracking.simulation` y `configs/benchmark.json`,
experimento 1.0. El generador usa la licencia MIT del proyecto. Las unidades
`u` son espaciales arbitrarias; un frame dura `dt=1`.

| Fuente | Referencia | Versión | Acceso | Condiciones |
|---|---|---|---|---|
| Generador sintético propio | `src/kalman_hungarian_object_tracking/simulation.py` | Experimento 1.0 y hashes en metadata | CLI simulate/benchmark | IDs ficticios, sin mediciones reales; CSV solo en outputs/ ignorado |

La verdad es `origen + frame * dt * velocidad`, con seis identidades estables.
Las trayectorias espaciadas y del cruce están en el JSON. `crossing`,
`dropout` y `clutter` comparten exactamente la geometría. No hay aceleraciones
verdaderas, reflexión ni recorte de posiciones.

Para cada semilla se usa `numpy.random.default_rng(seed)`: omisión Bernoulli,
ruido gaussiano independiente por coordenada y, solo en `clutter`, Poisson(2)
falsos positivos con posiciones uniformes en [0,100]². `dropout` añade la
ausencia de GT 0 en frames 57–62 inclusivos. Las detecciones se barajan antes
de asignar IDs crecientes, independientes de las identidades verdaderas.
El orden del consumo del RNG queda definido por el código y la configuración;
se verifica reproducción en el entorno registrado, sin prometer igualdad
entre todas las versiones de NumPy.

La verdad se guarda primero. El CSV de detecciones incluye `gt_id` para
evaluación y visualización; el tracker recibe registros inmutables con solo
`frame,detection_id,x,y`. Todos los métodos comparten los mismos registros.
Los datasets, incluso los sintéticos y los de tests, nunca se incorporan a Git.

Referencias de consulta, sin adquisición de datos:

- [SciPy: asignación lineal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).
- [SciPy: chi-cuadrado](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html).
- [MOTChallenge](https://motchallenge.net/), para distinguir sus métricas oficiales de este score propio.
- [Matplotlib: PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html).
