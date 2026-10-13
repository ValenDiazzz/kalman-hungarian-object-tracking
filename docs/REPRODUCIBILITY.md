# Reproducibilidad

La base requiere Python 3.11 o posterior. `pyproject.toml` declara el paquete y las
dependencias de desarrollo; no hay dependencias científicas ni experimentos.

## Ambiente y verificaciones de la base

Crear `.venv/`, activar el entorno e instalar `.[dev]` como indica el README. Para
comprobar importación y política de archivos:

```bash
python -m pytest
python scripts/check_staged_paths.py
```

Las pruebas de política también pueden ejecutarse con la biblioteca estándar:

```bash
python -m unittest discover -s tests -p test_staged_paths.py
```

Antes de instalar los hooks, asegurar que Gitleaks v8.30.0 está en `.venv/bin/`
o en `PATH`. `scripts/run_gitleaks.py` busca primero la instalación local y no
requiere activar el entorno virtual para ejecutar el hook. Mantener
`PRE_COMMIT_HOME` en `outputs/pre-commit-cache` para guardar su caché dentro de una
ruta ignorada. Verificar configuración e instalar los hooks en cada clon.

Los rangos de dependencias facilitan el arranque. Al implementar experimentos,
registrar las versiones resueltas de Python y bibliotecas y definir un ambiente
fijado antes de afirmar reproducción. Un inventario local puede obtenerse sin
exponer la ruta editable del proyecto:

```bash
mkdir -p outputs
python -m pip list --format=freeze > outputs/dependency-versions.txt
```

Este inventario no es un lockfile ni incluye todos los aspectos del cómputo.

## Convenciones para el desarrollo posterior

- Declarar explícitamente cada semilla en la configuración de ejecución. Aplicar
  semillas separadas a Python, NumPy y frameworks solo cuando esas dependencias
  se incorporen. No se establece ahora una semilla experimental ficticia.
- Guardar configuración pública con rutas relativas, parámetros, unidades y
  versiones de fuentes. Mantener credenciales y configuración local en archivos
  ignorados; no registrar variables de entorno completas.
- Documentar por ejecución el comando, la revisión del código, la configuración,
  las semillas, versiones y características de cómputo relevantes sin usernames,
  hostnames, rutas absolutas ni otros identificadores personales.
- Describir operaciones no deterministas y límites de repetibilidad. Separar
  pruebas de corrección, validaciones numéricas y comparaciones empíricas.
- Definir métricas, baselines, particiones y criterios de aceptación antes de
  ejecutar comparaciones. No mezclar datos de evaluación y ajuste.
- Crear scripts experimentales solo cuando exista un protocolo experimental definido. Registrar
  comandos que realmente se ejecutaron y distinguir pendientes de resultados.

## Datos y salidas

Obtener o generar datos únicamente en ejecución en `data/` o `datasets/` ignorados.
Las pruebas futuras que necesiten datos los crearán durante su ejecución en una
ruta ignorada, por ejemplo bajo `outputs/test-tmp/`; nunca agregar fixtures de
datasets al repositorio.

Guardar salidas, logs, métricas por muestra, ambientes e informes privados bajo
`outputs/` u otras rutas ignoradas. No agregar archivos de resultados de ejemplo.
Una figura o resumen agregado solo puede proponerse para Git después de revisar
procedencia, permisos, privacidad y correspondencia con resultados ejecutados.

## Pendientes científicos

Problema e interfaces, fuentes, configuración experimental, semillas, comandos,
métricas, baselines y resultados: **a completar al implementar**. Esta preparación
no verifica propiedades de ningún algoritmo.
