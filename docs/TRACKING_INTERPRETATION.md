# Interpretación científica de Kalman + Hungarian

## Alcance y evidencia examinada

El análisis usa la implementación de la revisión `3625944` y los resultados
guardados del experimento 1.0: 4 escenarios, 20 semillas `[100..119]`, cuatro
métodos y 320 corridas de 120 frames.

Fuentes locales: `outputs/benchmark/{assignments.csv,track_states.csv,trials.csv,
summary.csv,metadata.json,config_snapshot.json}` y los CSV de `outputs/data/`.
Se verificaron los hashes de código, configuración, datos y salidas registrados
en metadata. Las métricas se recalcularon sobre las 320 tablas guardadas;
el F1 se contrastó además con una construcción separada de la contingencia, y
los switches con un recuento independiente de transiciones.

Los CSV no guardan costos, velocidades ni covarianzas. Para examinarlos se hizo
una **reconstrucción del seguimiento**, a partir de las observaciones guardadas,
de siete secuencias:
los cuatro métodos de `separated/100`, los dos NN de `crossing/101` y
`kf_hungarian` de `clutter/100`. En sus 120 frames se comprobaron IDs,
nacimientos, posiciones de estado y flags contra las trazas existentes
(posiciones con tolerancia `1e-12`). Se usó una copia de los tracks para
reconstruir la predicción previa a la asociación, sin predecir dos veces en
el tracker original. No se registraron estas reproducciones como nuevos trials.

Los detalles y auxiliares de análisis quedan ignorados en
`outputs/interpretation/`. Este documento contiene explicaciones, conteos
agregados y costos derivados; no copia datasets ni coordenadas observadas.
Los ejemplos son diagnósticos: `separated/100` muestra alternancia sin cruces;
`crossing/101` es la primera semilla, en orden, donde NN-Hungarian obtiene peor
F1 que NN-greedy; `clutter/100` muestra competencia con tracks originados en FP.
La selección no cambia la evaluación ni pretende representar todas las semillas.

## 1. Definición matemática de detected_id_f1

La función responsable es
[`metrics.evaluate_associations`](../src/kalman_hungarian_object_tracking/metrics.py#L23).
Para una corrida, sean `g` una identidad verdadera, `p` un track y `d` una
detección observada. Se construye:

```text
M[g,p] = suma_d 1{d.gt_id=g y d.track_id=p}
```

Solo las detecciones con ambas identidades presentes contribuyen a M. Se busca
un matching de identidades uno-a-uno A que maximice `sum_(g,p)∈A M[g,p]`.
El código completa la matriz rectangular con ceros y usa
`linear_sum_assignment(..., maximize=True)`. Es un matching **sobre toda la
secuencia para puntuar**, distinto de las asociaciones online por frame.
No modifica los tracks y no proporciona GT al tracker.

```text
IDTP = máximo_A suma_(g,p)∈A M[g,p]
N_gt_det = número de detecciones con gt_id no nulo
N_pred_det = número de detecciones con track_id no nulo, incluidos FP
IDFN = N_gt_det - IDTP
IDFP = N_pred_det - IDTP
detected_id_f1 = 2 IDTP / (N_gt_det + N_pred_det)
               = 2 IDTP / (2 IDTP + IDFN + IDFP)
```

Si ambos totales son cero se devuelve NaN. En este benchmark cada detección
recibe un track, por asociación o nacimiento. Por ello
`N_pred_det=N_gt_det+false_positive_detections`; sin FP, el score se reduce
a `IDTP/N_gt_det`. Los IDs de los tracks no deben coincidir numéricamente con
los IDs GT: renombrarlos consistentemente no cambia el matching ni el F1.

**Ejemplo: separated/100.** Hay 706 detecciones verdaderas para 720
objeto-frames posibles y ningún FP. Ambos NN alcanzan IDTP=706 y F1=1, aunque
el detector omitió 14 observaciones y su recall es `706/720=0.980556`.
Ambos KF tienen IDTP=652, IDFN=54, IDFP=54 y F1=`652/706=0.923513`.
Aquí `IDFP=54` no significa 54 detecciones espurias: son observaciones asignadas
a identidades que no pertenecen al matching global elegido.

**Ejemplo: clutter/100, kf_hungarian.** Hay 645 detecciones verdaderas y
236 espurias; N_pred_det=881, IDTP=424, IDFN=221 e IDFP=457. Entonces
F1=`848/1526=0.555701`. Con esos mismos totales, incluso IDTP=645 daría
un máximo de `1290/1526=0.845347`, por el efecto de los FP sobre el denominador.

### Diferencia con el IDF1 estándar

La forma algebraica y el matching global uno-a-uno son compatibles con la
familia de métricas de identidad de Ristani et al.; la diferencia principal
está en **qué observaciones se evalúan y qué constituye una coincidencia**.
En el IDF1 estándar se contabilizan las anotaciones GT elegibles y las salidas
del tracker en toda la secuencia, y la compatibilidad depende de un criterio
espacial. En MOT 2D normalmente se usan cajas y un umbral de IoU, no la etiqueta
de origen de una detección. Véanse la [formulación original, §§3.3–3.4](https://arxiv.org/html/1609.01775v2)
y [`TrackEval.Identity.eval_sequence`](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/identity.py).

| Aspecto | detected_id_f1 del proyecto | IDF1 del protocolo MOT 2D |
|---|---|---|
| Universo GT | Detecciones verdaderas presentes | Anotaciones GT elegibles, también sin detección del detector |
| Compatibilidad | `gt_id` de origen incorporado después de tracking | Geometría de GT y predicción, con umbral espacial |
| Error de localización | No interviene directamente en el scoring | Puede impedir una coincidencia espacial |
| Estados durante misses | No entran al score: no son asociaciones a detecciones | Su efecto depende de si se exportan como predicciones del tracker |
| Detección espuria cerca de un objeto | Conserva gt_id nulo y penaliza N_pred_det | Se examina según geometría y reglas del protocolo |

La referencia de [TrackEval para MOT 2D](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/datasets/mot_challenge_2d_box.py)
calcula la similitud con IoU de cajas; `Identity` tiene umbral 0.5 por defecto.
Esto no está implementado en el evaluador de puntos sintéticos del proyecto.

Las ausencias del detector no añaden IDFN aquí. `n_gt_total` se usa únicamente
para `detection_recall`, no como denominador del F1. No basta sustituir 706
por 720 para obtener IDF1 estándar: también cambiarían las coincidencias
espaciales y la definición de las salidas evaluadas. No se ejecutó TrackEval;
no hay un valor IDF1 oficial calculado para estas simulaciones.

## 2. Cómo se contabilizan id_switches

En [`evaluate_associations`](../src/kalman_hungarian_object_tracking/metrics.py#L23)
se toman las filas con GT no nulo, se ordenan por frame/detection_id y se
recorre cada objeto por separado. Si sus identidades observadas son
`p_g,1,...,p_g,n`, se calcula:

```text
id_switches = suma_g suma_k=2..n_g 1{p_g,k != p_g,k-1}
id_switches_per_100_detections = 100 * id_switches / N_gt_det
```

No cuenta el primer avistamiento. Una alternancia A→B→A cuenta dos switches;
un cambio permanente A→B cuenta uno, independientemente de cuántas detecciones
permanezcan luego en B. Un nacimiento posterior cuenta si cambia la identidad.
Los FP no tienen GT sobre el que contar switches, aunque sí pueden influir en
las asociaciones de objetos verdaderos. Si no hay detecciones verdaderas, el
recuento es cero y la tasa normalizada es NaN.

El predecesor es la **detección anterior observada**, no necesariamente el frame
anterior. En `dropout/100`, GT 1 pasa de T5 en frame 110 a T8 en frame 112;
la ausencia de observación en 111 no evita que se cuente un switch. En la misma
corrida, GT 0 conserva T4 entre sus observaciones de frames 56 y 64: esa ausencia
no cuenta como switch por sí sola.

El código transforma un track_id nulo en `None` antes de comparar; una transición
de ID a None o viceversa contaría. Este caso no aparece en el benchmark, porque
toda detección recibe identidad. Es relevante si se reutiliza la métrica con
salidas parciales.

No se implementa una evaluación CLEAR MOT: no se hace matching espacial contra
GT por frame. Tampoco sería correcto afirmar que contar a través de huecos es
una diferencia exclusiva con CLEAR: [TrackEval.CLEAR.eval_sequence](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/clear.py)
también recuerda el último ID asociado para puntuar IDSW, separándolo del
matching del frame anterior. Aquí la diferencia es el universo observado, la
correspondencia de origen y la normalización definida por el proyecto.

## 3. Por qué aparecen más switches con Kalman

### Lo que muestran las veinte semillas

Medianas de switches por 100 detecciones verdaderas, desde `summary.csv`:

| Escenario | nn_greedy | nn_hungarian | kf_greedy | kf_hungarian |
|---|---:|---:|---:|---:|
| separated | 0.707550 | 0.707550 | 7.357814 | 7.357814 |
| crossing | 1.281466 | 1.199080 | 8.226363 | 7.943077 |
| dropout | 0.780644 | 0.703268 | 5.585890 | 5.585890 |
| clutter | 5.369428 | 5.417957 | 26.732077 | 21.826678 |

Se clasificó cada switch según si **ambos tracks recibieron detecciones verdaderas
de un único y mismo GT durante toda su vida en la corrida**. Esta clasificación
usa etiquetas solo para analizar las trazas; no es una nueva métrica de tracking
ni modifica las ya reportadas. Puede coexistir con observaciones FP en clutter.

| Escenario | Método | Switches totales en 20 semillas | Entre tracks exclusivos del mismo GT |
|---|---|---:|---:|
| separated | nn_greedy | 195 | 195 |
| separated | nn_hungarian | 195 | 195 |
| separated | kf_greedy | 1160 | 1132 |
| separated | kf_hungarian | 1160 | 1132 |
| crossing | nn_greedy | 252 | 121 |
| crossing | nn_hungarian | 216 | 121 |
| crossing | kf_greedy | 1219 | 757 |
| crossing | kf_hungarian | 1197 | 782 |
| dropout | nn_greedy | 160 | 79 |
| dropout | nn_hungarian | 153 | 77 |
| dropout | kf_greedy | 933 | 769 |
| dropout | kf_hungarian | 939 | 793 |
| clutter | nn_greedy | 694 | 510 |
| clutter | nn_hungarian | 672 | 501 |
| clutter | kf_greedy | 3403 | 2504 |
| clutter | kf_hungarian | 2886 | 2120 |

En separated, 97.586% de los switches KF no necesitan confusión entre dos objetos
verdaderos para explicarse: alternan tracks del mismo objeto. No son todos
intercambios durante cruces. Solo 71 de los 1160 switches de cada KF coinciden
con el nacimiento del track entrante; los otros 1089 entran a un track existente.
NN también puede tener duplicados; el mecanismo no es exclusivo de KF.

### Mecanismo observado: rechazo, nacimiento inmediato y duplicación

[`Tracker.step`](../src/kalman_hungarian_object_tracking/tracker.py#L68)
crea un track por cada detección no asignada y conserva el track sin match
hasta superar ocho misses consecutivos. No hay confirmación ni supresión de
duplicados. El caso `separated/100`, GT 5, permite seguir la cadena:

- Frame 29: la detección se asocia a T1.
- Frame 30: T1 tiene costo Mahalanobis² 11.989407, superior a 9.21034.
  La arista se prohíbe; ningún track KF admite esa detección y nace T6.
  T1 permanece activo con un miss.
- En el mismo frame y para la misma detección, NN-T1 tiene costo euclídeo²
  16.225302 <= 25 y conserva la asociación. Las referencias de posición
  difieren entre familias; no se atribuye todo el contraste solo a la puerta.
- Frame 32: la detección vuelve a T1, mientras T6 acumula un miss.
- Frame 36: vuelve a T6; el análisis de covarianza siguiente muestra por qué.

Al terminar, ambos KF asignaron las 118 detecciones de GT 5 entre T1 (88) y
T6 (30), con **26 switches**; ambos NN conservaron T1 para las 118 y tuvieron
cero switches de ese objeto. T1 y T6 nunca recibieron otro GT verdadero.
La pureza global de la corrida KF es 1, pese a sus 57 switches y F1=0.923513:
pureza no equivale a continuidad ni a identidad única por objeto.

### Mecanismo observado: la incertidumbre cambia la competencia entre tracks

[`association.kf_costs`](../src/kalman_hungarian_object_tracking/association.py#L23)
usa `rᵀS⁻¹r`, con S de
[`KalmanFilter.innovation_covariance`](../src/kalman_hungarian_object_tracking/kalman.py#L63).
En este experimento las covarianzas de innovación son isotrópicas,
`S=s I₂`: el costo es `distancia_a_predicción²/s` y la puerta equivale a un
radio `sqrt(9.21034*s)`. Mayor incertidumbre reduce el costo para un mismo
residual y amplía la región admisible.

En `separated/100`, GT 5, **frame 36**, se reconstruyeron:

| Track | Misses antes del frame | Distancia² a predicción (u²) | s (u²) | Mahalanobis² | Asociación emitida |
|---|---:|---:|---:|---:|---|
| T1 | 0 | 4.509615 | 1.397778 | 3.226273 | Sin detección |
| T6 | 4 | 21.444536 | 62.030908 | 0.345707 | Recibe la detección de GT 5 |

Ambas aristas son admisibles. T6 está más lejos en distancia euclídea, pero
tiene menor costo Mahalanobis por su S mucho mayor. Esto identifica el
mecanismo de esa decisión concreta, no demuestra que la misma causa explique
cada switch del benchmark. T6 conserva una velocidad reconstruida
`(0.302228,-0.885433) u/frame`, frente a la velocidad verdadera constante
`(0.1,0.03)`: tras pocas observaciones ruidosas su predicción se aleja, sin que
la gran covarianza le impida ganar la asociación.

El crecimiento proviene de
[`KalmanFilter.predict`](../src/kalman_hungarian_object_tracking/kalman.py#L53):
`P⁻=F P Fᵀ+Q`. Durante misses no hay update que reduzca P. El nacimiento
inicializa velocidad cero y `P0=diag(4,4,4,4)`;
[`initialize`](../src/kalman_hungarian_object_tracking/kalman.py#L31) da al
track nuevo una incertidumbre de velocidad elevada. En frame 31, T6 tiene
`S=9.000625 I₂`, frente a `1.506647 I₂` de T1. Los matches reinician el contador
de misses, de modo que los dos tracks pueden sobrevivir alternándose.

Las puertas tampoco son equivalentes entre NN y KF. En frame 30 el radio KF
de T1 era aproximadamente 3.554416 u, frente al radio fijo NN de 5 u; para T6
en frame 36 llega a 23.902421 u. Por tanto, cambiar de familia también cambia
qué detecciones se rechazan o se aceptan. No es un ensayo que aísle únicamente
el beneficio de predecir movimiento.

El costo implementado **no contiene log(det S)**. Si se compararan densidades
gaussianas normalizadas, su negativo logarítmico incluiría
`0.5*(rᵀS⁻¹r + log(det S))`, además de constantes y los supuestos de asociación
que correspondan. El objetivo de asociación usa únicamente el término cuadrático.
La ventaja para un candidato de alta incertidumbre queda demostrada en el
ejemplo; no se ha probado que añadir otro término mejore el tracking.

### Evidencia adicional en clutter

En `clutter/100`, `kf_hungarian`, T8 nace de una detección espuria en frame 0.
En frame 5, GT 0 cambia de su anterior T4 a T8:

| Track | Origen | Misses antes del frame | Distancia² a predicción (u²) | s (u²) | Mahalanobis² |
|---|---|---:|---:|---:|---:|
| T4 | Detección verdadera | 0 | 8.296571 | 2.284198 | 3.632160 |
| T8 | Falso positivo | 4 | 269.476906 | 105.103125 | 2.563929 |

Ambos candidatos pasan la puerta. El track originado en FP tiene menor costo
aun estando mucho más lejos; recibe GT 0 y T4 queda sin match. La sustitución
T8→T4 para esa detección mantendría el número de misses y sería más costosa.
Es evidencia concreta de cómo clutter puede introducir competidores de alta
incertidumbre; no una estimación causal del efecto de todos los FP.

Sobre veinte semillas, 2723 de los 2886 switches de KF-Hungarian en clutter
involucran al menos un track que recibió algún FP durante su vida. El criterio
mira toda la vida del track: un FP posterior también cumple la condición.
Por ello **ese conteo no permite afirmar que 2723 switches fueron causados por
FP**. En particular, alternancia entre tracks del mismo GT y contacto con FP
no son categorías excluyentes.

### Explicaciones respaldadas y límites de causalidad

| Afirmación | Estado de la evidencia |
|---|---|
| Un rechazo puede crear un duplicado que coexiste con el track anterior | Demostrado por código y separated/100, frame 30 |
| Un track con alta S puede ganar pese a estar más lejos | Demostrado por los costos reconstruidos de frames 36 y clutter/100 frame 5 |
| Hay muchos switches sin mezcla de dos GT verdaderos | Demostrado por clasificación de todas las trazas guardadas |
| Los cruces y FP contribuyen a ambigüedad | Compatibilidad de geometría y ejemplos concretos; no descomposición causal completa |
| La inicialización o sigma_a están mal elegidos | No establecido: requeriría calibración y comparaciones controladas |
| Q/R mal calibrados causan todo el exceso de switches | Hipótesis no probada; no se midió una distribución de innovaciones ni se hizo una ablation |
| Añadir confirmación, suprimir duplicados o penalizar incertidumbre mejoraría F1 | Hipótesis para otro protocolo; no se implementó ni evaluó aquí |
| Los resultados demuestran que Kalman es peor en general | No sustentado: se comparan pipelines con predictor, costo y puerta diferentes |

No hay aceleraciones verdaderas, rebotes ni maniobras en
[`simulation.ground_truth`](../src/kalman_hungarian_object_tracking/simulation.py#L16).
No corresponde explicar estos resultados por movimientos industriales no
modelados. Tampoco un rechazo por puerta χ² demuestra, por sí mismo, un bug o
sobreconfianza: una puerta del 99% permite rechazos incluso bajo su modelo.
La frecuencia y calibración de las innovaciones no se auditaron estadísticamente.

## 4. Por qué Hungarian puede tener peor F1 que Greedy

[`augmented_cost_matrix`](../src/kalman_hungarian_object_tracking/association.py#L79)
y [`hungarian_assignment`](../src/kalman_hungarian_object_tracking/association.py#L97)
minimizan, en cada frame y con los estados de ese método:

```text
J(A) = suma_(i,j)∈A C_ij
       + lambda * (tracks sin detección + detecciones sin track existente)
lambda = gate_cost + 1
```

Las aristas prohibidas no son candidatas. Las detecciones sin track existente
originan nacimientos después de resolver. NN usa lambda=26 u²; KF usa
lambda=10.21034 en costo adimensional. El solver optimiza **ese J**,
no detected_id_f1, switches, pureza ni continuidad futura.
[SciPy documenta el óptimo de la matriz de costos suministrada](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).

La función
[`greedy_assignment`](../src/kalman_hungarian_object_tracking/association.py#L59)
ordena todos los pares admisibles por `(costo,track_id,detection_id)` y toma
extremos libres. No tiene garantía de mínimo global. Sin embargo, una decisión
greedy más costosa en el frame puede terminar conservando una identidad durante
más observaciones. Las decisiones cambian las referencias del frame siguiente;
tras divergir los métodos, sus matrices ya no tienen por qué ser iguales.

### Ejemplo real: crossing/101

Antes del frame 59, ambos NN tenían las mismas posiciones y asociaciones:
GT 0 usaba T6 y GT 1 usaba T2. Para las dos detecciones del cruce:

| Track | Costo a detección de GT 0 (u²) | Costo a detección de GT 1 (u²) |
|---|---:|---:|
| T2 | 1.932106 | 3.144703 |
| T6 | 4.983396 | 7.118516 |

Los cuatro pares pasan la puerta. Greedy toma T2→GT0 y T6→GT1, suma
9.050622; Hungarian toma T2→GT1 y T6→GT0, suma 8.128099. Los otros cuatro
matches no cambian y no hay penalidades por misses/nacimientos en ese frame.
El objetivo completo es 27.938730 frente a 27.016207: Hungarian obtiene el
menor costo y **en ese frame conserva las identidades previas**. No sería
correcto presentar esta primera decisión global como un intercambio incorrecto.

La evolución observada después es distinta:

- NN-greedy intercambia las dos identidades durante las observaciones de frames
  59–60 y vuelve a la correspondencia anterior en frame 61.
- NN-Hungarian conserva la correspondencia en 59–60, pero en frame 61 la
  intercambia y la mantiene hasta el final en las detecciones presentes.
- En frame 61, greedy y LSAP calculados sobre los mismos estados de Hungarian
  producen el mismo emparejamiento. El cambio persistente no requiere que el
  solver falle ni que el óptimo de ese frame difiera de greedy.

Por tanto, no se atribuye todo el deterioro a una única elección local errónea
en frame 59. Se observa una trayectoria de decisiones y actualizaciones bajo
un objetivo espacial que no reconoce la identidad verdadera ni su duración.

Los conteos agregados de las dos identidades que se cruzan son:

| Método | GT 0: detecciones por track | GT 1: detecciones por track |
|---|---|---|
| nn_greedy | T1:23, T2:2, T6:93 | T2:116, T6:2 |
| nn_hungarian | T1:23, T2:58, T6:37 | T2:60, T6:58 |

Hay una fragmentación previa compartida: GT 0 pasó de T1 a T6 en frame 23.
Para greedy, el matching global de evaluación elige GT0↔T6 y GT1↔T2,
aportando 93+116=209 coincidencias. Para Hungarian elige GT0↔T2 y GT1↔T6,
aportando 58+58=116. Los otros cuatro objetos aportan 471 en ambos casos:

| Método | IDTP | IDFN=IDFP | F1 | Switches |
|---|---:|---:|---:|---:|
| nn_greedy | 680 | 27 | 0.961810 | 5 |
| nn_hungarian | 587 | 120 | 0.830269 | 3 |

El ejemplo muestra por qué **menos switches no implica mayor F1**: la duración
de las asociaciones incompatibles con un matching global es distinta. No
hay contradicción entre la optimalidad por frame y el peor score de secuencia.

La variación entre semillas también importa. En crossing, NN-Hungarian tiene
mejor F1 que NN-greedy en 11 semillas, peor en 8 y empata en 1. Aun así, la
diferencia de medianas es −0.018945 y la media de diferencias pareadas es
−0.008909. La mediana de diferencias pareadas es +0.002817: **diferencia de
medianas y mediana de diferencias son cantidades distintas**. Ninguna de estas
cifras, por sí sola, constituye un test de significancia ni superioridad universal.

## 5. Diferencias concretas entre NN y KF

| Componente | Familia NN | Familia KF |
|---|---|---|
| Estado espacial | Última posición observada | Estado `[x,y,vx,vy]` y covarianza P |
| Antes de asociar | Conserva la observación anterior | Predice todos los tracks activos con F y Q |
| Costo | Distancia euclídea² a última observación | Mahalanobis² a predicción, usando S=H P⁻ Hᵀ+R |
| Puerta | Radio fijo 5 u; C<=25 u² | C<=9.21034; región espacial variable según S |
| Match | Reemplaza posición por la detección | Actualiza estado y P con solve y Joseph |
| Miss | Mantiene fija la última observación | Propaga posición, velocidad e incertidumbre; no inventa una medición |
| Nacimiento | Posición de la detección | Esa posición, velocidad cero y P0=diag(4,4,4,4) |
| Estrategia greedy | Mismo orden de pares admisibles | Mismo orden, pero costos/puertas diferentes |
| Estrategia Hungarian | Mismo esquema aumentado; lambda=26 | Mismo esquema aumentado; lambda=10.21034 |

Las reglas de identidad y supervivencia son compartidas en
[`Tracker.step`](../src/kalman_hungarian_object_tracking/tracker.py#L68): una
detección por track y viceversa, nacimiento inmediato por detección no asignada,
misses consecutivos, reset al hacer match, eliminación en el noveno miss,
IDs crecientes sin reciclaje y sin confirmación ni reidentificación por apariencia.
Ambas familias pueden mantener varios tracks históricos del mismo objeto.

El filtro usa el modelo de velocidad constante de
[`model_matrices`](../src/kalman_hungarian_object_tracking/kalman.py#L10) y
[`update`](../src/kalman_hungarian_object_tracking/kalman.py#L66), con dt=1,
sigma_det=1 y sigma_a=0.05. Son parámetros configurados, no estimados con
datos industriales.
[`experiments.observation_frames`](../src/kalman_hungarian_object_tracking/experiments.py#L38)
crea registros inmutables sin GT, compartidos entre métodos;
[`run_benchmark`](../src/kalman_hungarian_object_tracking/experiments.py#L93)
incorpora gt_id solo después del seguimiento. Las diferencias observadas no
provienen de dar distintas detecciones a cada tracker.

## Qué permiten concluir estas trazas

El exceso de switches KF tiene evidencia directa de duplicación y competencia
ponderada por incertidumbre, incluso sin cruces y con pureza 1 en un ejemplo.
Los cruces añaden ambigüedad de identidad y los FP pueden crear competidores;
no hay una descomposición causal completa ni una ablation de cada componente.
F1, switches y pureza deben leerse juntos: miden duración global, transiciones
y mezcla de GT por track, respectivamente, dentro de sus definiciones locales.

NN frente a KF cambia estimador, costo y puerta, y las penalizaciones expresadas
en sus respectivas unidades. Greedy frente a Hungarian dentro de cada familia
sí compara la estrategia de asociación con el resto del protocolo compartido.
No se infiere un ranking general de Kalman ni una garantía de conservación de
identidad a partir de un óptimo espacial instantáneo.

La verificación incluyó hashes y métricas sobre artefactos existentes, además
del contraste de las siete reconstrucciones seleccionadas con sus CSV.
