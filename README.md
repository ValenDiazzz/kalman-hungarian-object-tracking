# Kalman Hungarian Object Tracking

Base de un proyecto científico y educativo de Python para desarrollar trabajo
posterior con código revisable, reproducción documentada y protección de archivos
privados. El nombre identifica el repositorio; en esta etapa solo existen la
estructura, documentación y controles de seguridad.

**Estado:** preparación inicial. No hay algoritmos científicos implementados,
experimentos, datasets ni resultados. Los objetivos, interfaces, supuestos,
baselines y criterios de aceptación se definirán al iniciar el desarrollo.

## Instalación

Requisitos: Python 3.11 o posterior y Git. Desde la raíz del repositorio:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --no-cache-dir -e ".[dev]"
python -m pytest
```

La instalación de desarrollo incorpora `pytest` y `pre-commit`; el paquete no
requiere dependencias científicas. Los rangos de `pyproject.toml` definen
compatibilidad, pero todavía no constituyen un ambiente experimental fijado.

Para las comprobaciones de secretos, instalar Gitleaks **v8.30.0** desde su
[release oficial](https://github.com/gitleaks/gitleaks/releases/tag/v8.30.0),
comprobar el SHA-256 del artefacto contra los checksums de esa release y colocar
el ejecutable en `.venv/bin/`, que está ignorado, o en `PATH`. El hook busca
primero la instalación local, por lo que no requiere activar el entorno para
hacer un commit. Para comprobarla y configurar los hooks:

```bash
.venv/bin/gitleaks version
mkdir -p outputs/pre-commit-cache
export PRE_COMMIT_HOME="$PWD/outputs/pre-commit-cache"
.venv/bin/pre-commit validate-config
.venv/bin/pre-commit install
python scripts/check_staged_paths.py
```

El hook oficial `gitleaks-system` está fijado a `v8.30.0` en
`.pre-commit-config.yaml` y usa `scripts/run_gitleaks.py` para localizar el
ejecutable instalado. La ausencia de
Gitleaks, un fallo de Git o un fallo de conexión al preparar hooks debe corregirse;
no equivale a una comprobación aprobada. Las cachés locales quedan ignoradas.

## Estructura

```text
.gitignore
.pre-commit-config.yaml
pyproject.toml
README.md
docs/
  DATA_SOURCES.md
  REPRODUCIBILITY.md
  SECURITY.md
scripts/
  check_staged_paths.py
  run_gitleaks.py
src/
  kalman_hungarian_object_tracking/
    __init__.py
tests/
  test_smoke.py
  test_staged_paths.py
  test_gitleaks_hook.py
```

`outputs/`, los entornos, las cachés y las carpetas de datos se crean localmente
cuando sean necesarios y permanecen ignorados. Las pruebas actuales usan nombres
de rutas y placeholders; no incluyen datasets ni secretos reales.

## Reproducción: a completar al implementar

La comprobación de esta base es `python -m pytest`. Antes del primer experimento,
documentar en [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) el comando de ejecución,
configuración, semillas, versiones exactas, métricas y criterios de aceptación.
Los comandos experimentales y sus resultados están pendientes; no hay una CLI
científica disponible.

## Procedencia de datos

**SIN FUENTES DEFINIDAS.** [DATA_SOURCES.md](docs/DATA_SOURCES.md) contiene la
plantilla para registrar referencias oficiales, licencias, versiones y acceso.
Nunca se incorporan datasets a Git, incluidos datos públicos, sintéticos,
muestras y fixtures. Una futura descarga o generación ocurrirá en ejecución en
rutas ignoradas, después de verificar las condiciones de uso.

## Protección y revisión

`.gitignore` evita inclusiones habituales y `scripts/check_staged_paths.py`
rechaza rutas protegidas del índice, aunque se haya eludido el ignorado. El hook
revisa también que cada `.env.example` staged use únicamente asignaciones
ficticias, por ejemplo `VARIABLE=CHANGE_ME`, `VARIABLE=REPLACE_ME`,
`VARIABLE=EXAMPLE_ONLY` o `VARIABLE=`. Admite líneas vacías; no admite comentarios
ni valores arbitrarios. No hace falta crear un ejemplo de ambiente ahora.

Después de seleccionar archivos públicos para staging, ejecutar:

```bash
python scripts/check_staged_paths.py
.venv/bin/pre-commit run
git status --short
```

`pre-commit run` y Gitleaks en modo staged examinan el índice. En un índice vacío
no certifican el contenido de archivos nuevos sin seguimiento. Antes de publicar,
revisar también esos archivos y el historial, siguiendo
[SECURITY.md](docs/SECURITY.md). Los hooks se instalan por clon y se pueden eludir;
complementan la revisión humana.

## Limitaciones y licencia

No se puede inferir desempeño, corrección matemática ni utilidad práctica de esta
estructura inicial. No hay evidencia experimental para comparar métodos. La
reproducción científica y las dependencias necesarias se completarán cuando haya
un protocolo experimental definido. La licencia del proyecto debe elegirse antes de su
publicación; no se asigna una licencia ni un contacto privado por suposición.
