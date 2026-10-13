# Dependencias y licencias

El código del proyecto usa la licencia MIT. No se
redistribuyen datasets, modelos ni binarios de dependencias. Las bibliotecas
se instalan por pip; mantener sus avisos y revisar licencias de componentes
incluidos si se distribuyen binarios o un entorno empaquetado.

| Dependencia directa | Versión ejecutada | Licencia principal / referencia oficial |
|---|---|---|
| NumPy | 2.5.4 | [BSD-3-Clause y componentes adicionales](https://github.com/numpy/numpy/blob/main/LICENSE.txt); metadata incluye 0BSD, MIT, Zlib y CC0-1.0 |
| SciPy | 1.18.1 | [BSD-3-Clause y avisos de terceros](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| pandas | 3.0.6 | [BSD-3-Clause](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| Matplotlib | 3.11.2 | [Licencia basada en PSF y componentes compatibles BSD](https://matplotlib.org/stable/project/license.html) |
| Pillow | 12.3.0 | [MIT-CMU](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| pytest (dev) | 9.1.1 | [MIT](https://github.com/pytest-dev/pytest/blob/main/LICENSE) |
| pre-commit (dev) | 4.7.0 | [MIT](https://github.com/pre-commit/pre-commit/blob/main/LICENSE) |
| Ruff (dev) | 0.17.0 | [MIT](https://github.com/astral-sh/ruff/blob/main/LICENSE) |
| Gitleaks (herramienta local) | 8.30.0 | [MIT](https://github.com/gitleaks/gitleaks/blob/master/LICENSE) |

Las versiones y etiquetas anteriores se contrastaron con metadata del entorno
y referencias públicas. El backend setuptools se declara en pyproject.toml;
se instala en aislamiento de build y no es una dependencia científica runtime.
requirements-reproducible.txt registra también dependencias transitivas del
entorno validado, sin rutas editables ni identificadores personales. No es
un lock con hashes de wheels ni una auditoría de todas las posibles
distribuciones binarias/plataformas. Las referencias main/stable pueden cambiar;
consultar los avisos incluidos en la distribución exacta si se redistribuye.
