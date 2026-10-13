# Contratos matemáticos y pruebas

El estado es `[x,y,vx,vy]`, unidades `u` y `u/frame`. Los costos NN son `u²`;
Mahalanobis² es adimensional. Las puertas son inclusivas. Todas las asociaciones
son uno-a-uno por frame. El estimador nunca recibe etiquetas verdaderas.

| Invariante / objetivo | Prueba | Oráculo o control |
|---|---|---|
| Geometría fija y cruce en frame 60 | `test_required_geometries` | Posiciones analíticas, sin RNG |
| Reproducibilidad de simulación | `test_reproducible_truth_and_detector` | Igualdad de tablas; verdad independiente de semilla |
| Ventana dropout y FP exclusivos de clutter | `test_window_and_clutter_only_in_prescribed_scenarios` | Detector con omisión aleatoria nula |
| Matrices F, H, Q, R y orientación | `test_model_matrices_and_constant_velocity` | G Gᵀ y predicción analítica con dt=2 |
| Ganancia y actualización | Tests analíticos de update | Cálculo escalar independiente, incluidas covarianzas cruzadas |
| P finita, simétrica y PSD durante misses | `test_joseph_covariance_psd_finite_and_symmetric_after_misses` | Autovalores; ruido 1e-4, 1, 100 y 101 predicciones sin update |
| Mahalanobis y puerta | `test_mahalanobis_cost_and_chi_square_gate` | Ejemplo diagonal y `chi2.ppf(.99,2)` |
| Óptimo del objetivo aumentado | `test_lsap_optimality_with_penalties_and_gating` | Todos los matchings parciales para 0–3 extremos |
| Aristas prohibidas | `test_invalid_edges_are_removed_before_optimization` | Contraejemplo con asignación bloqueada más barata |
| Greedy sin garantía global | `test_augmented_blocks_and_greedy_counterexample` | Costos manuales 25 frente a 4 |
| Penalización distinta de maximizar cardinalidad | `test_unmatched_penalties_can_prefer_lower_cardinality` | Dos matches + penalidades 52 frente a tres de costo 75 |
| Nacimientos, misses, muerte, IDs crecientes | `test_birth_misses_reset_death_and_no_recycled_ids` | Secuencia manual con max_missed=2 |
| NN retiene posición; KF predice en misses | Tests de posición/misses | Estado previo y velocidad, sin pseudoobservaciones |
| Exclusión de GT | Tests de contrato y stream compartido | Registros con slots y campos exactos; inspección de módulos |
| F1, switches, FP, NaN y permutaciones | `test_metrics.py` | Conteos manuales y matching de identidades por fuerza bruta |
| CLI, imágenes y repetición | `test_clean_cli_outputs_and_deterministic_repetition` | Subprocesos sin PYTHONPATH, Pillow y comparación de outputs |
| Benchmark completo y segunda ejecución | `scripts/verify_benchmark.py` | Recalcular scores, hashes y resumen; 320/16 filas y GIF completo |

SciPy LSAP minimiza el objetivo aumentado por frame, sin certificar continuidad
óptima en el tiempo. Greedy es una heurística. El filtro usa `solve`,
simetrización y Joseph. El control runtime permite autovalores de redondeo
hasta `-1e-10 * max(1, ||P||₂)`; los tests usan `-1e-10` en sus escalas.
La puerta redondeada 9.21034 se contrasta con tolerancia absoluta 1e-6.
No se afirma estabilidad para toda escala ni bajo covarianzas mal especificadas.

`max_missed=8` permite ocho misses consecutivos y elimina en el noveno
(`missed > max_missed`). Un match reinicia el contador; todo nacimiento se
reporta inmediatamente, sin confirmación. Los duplicados de un objeto pueden
coexistir y alternar su identidad. El tracker no incluye una etapa de
confirmación ni supresión de duplicados.
