# Seguridad y privacidad

Tratar cada archivo potencialmente publicable como público desde su creación.
Nunca incorporar a Git secretos, claves privadas, credenciales, información
personal, material privado de terceros, configuraciones locales ni notas internas.
Tampoco incorporar datasets de ninguna clase, incluidos públicos, sintéticos
y usados en tests. Registrar fuentes y condiciones en `DATA_SOURCES.md`;
guardar datos y salidas de ejecución en rutas ignoradas.

## Controles locales

`.gitignore` limita las carpetas raíz y los documentos Markdown a las rutas
del proyecto; excluye configuraciones ocultas, datos, ambientes, claves, logs,
respaldos y formatos habituales de datos y pesos. El hook local aplica la misma
lista de rutas e inspecciona los archivos staged con Git. Rechaza las rutas
protegidas sin imprimir nombres ni contenidos. Inspecciona el blob
staged de `.env.example` y acepta únicamente placeholders fijos. Gitleaks añade
escaneo de secretos con el hook oficial fijado a v8.30.0. Al agregar una carpeta
raíz o documentación, actualizar las listas de `.gitignore` y
`scripts/check_staged_paths.py`.

Una ausencia de herramientas o un error de ejecución no debe reportarse como
aprobación. No desactivar silenciosamente hooks por falta de conexión. Instalar
los hooks por clon, revisar los archivos sin seguimiento y comprobar el índice:
el ignorado no elimina archivos previamente rastreados y los hooks son eludibles.

La política de rutas no puede reconocer datasets con extensiones arbitrarias,
datos privados incrustados en código, imágenes o documentación, ni todas las
formas de secretos. Revisar manualmente contenido, procedencia, metadatos y
licencias antes de incorporarlos. No usar `git add -f` para saltar exclusiones.

## Antes de publicar

Revisar status, rutas rastreadas y staged, reglas de ignorado, tests de política,
escaneo de secretos y material sin seguimiento. Ejecutar Gitleaks con redacción
sobre los archivos candidatos; excluir ambientes y material privado local del
alcance publicable. Guardar informes solo en `outputs/` ignorado.

El modo staged se ejecuta con:

```bash
gitleaks git --pre-commit --staged --redact
```

Cuando exista historial, revisar ramas y tags pertinentes y ejecutar:

```bash
gitleaks git --redact .
```

Sin commits, el escaneo de historial no aplica. Una comprobación limpia de un
índice vacío no certifica archivos sin seguimiento. Antes del primer push se
debe repetir la auditoría sobre el contenido que realmente se pretenda publicar.
Verificar además la licencia MIT del proyecto y las licencias de fuentes y
artefactos externos.

## Reporte privado de vulnerabilidades

Todavía no existe un contacto privado ni un canal de reporte establecido. Antes
de publicar, definir un canal verificable. Si se habilita el reporte privado de
vulnerabilidades en GitHub, usar la opción correspondiente de la pestaña Security
del repositorio público. Mientras no exista ese canal, no publicar detalles
sensibles en issues, discusiones ni pull requests; solicitar al mantenedor un
medio privado sin incluir la evidencia sensible en la solicitud.

Ante una exposición, detener la publicación, informar sin revelar el valor del
secreto y coordinar revocación o rotación. La eliminación de un archivo no limpia
el historial: cualquier reescritura o corrección destructiva requiere autorización
expresa. No copiar evidencias sensibles a logs, documentación o informes públicos.
