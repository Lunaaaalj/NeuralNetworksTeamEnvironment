# Cierre de alineación A3/A4

26 de octubre de 2026. Rama `fix/avance4-alineacion-avance3`, base `fe49c37` de `origin/main`. Complementa el cierre anterior de `correcciones_etapa5/CIERRE.md`, conservado como auditoría histórica.

## Cambios y evidencia

- Matriz de trazabilidad en `ALINEACION_AVANCE3_AVANCE4.md`: problema/datos, arquitectura, familias, origen de MLP, regularización, épocas/pérdidas, selección y límites.
- Resumen, objetivos, contexto, dataset, metodología, protocolo, regularización y conclusiones conectan A3/A4. Se conservan las ocho secciones académicas, tablas, curvas, anexos y resultados.
- La metodología incluye «Continuidad experimental respecto al Entregable 3» y una tabla breve de procedencia. Se distingue evolución académica de orden literal de ejecución.
- CNN compartida: 480,936 parámetros; A3 Adam/batch 32, checkpoint 38/43 y F1 0.5538; A4 original DirectML/batch 64, checkpoint 53/58 y F1 0.5740. Son distintas ejecuciones; no se afirma continuidad de pesos ni mejora por arquitectura.
- MLP reconstruida posteriormente: 9,660,776 parámetros, Adam 1e-4, dropout 0.5, batch 64, checkpoint 77/87, F1 0.0280 y exactitud 15.52%; bajo ajuste y comparación limitada por política/capacidad. A3 no fue una MLP.
- Ocho archivos summary/history idénticos en cuatro alias A3/F: no aumentan n. La regularización ya documentada en A3 se analiza con mayor detalle, conservando checkpoints y época común 34. La intervención conjunta no aísla técnicas; LR difiere en la época común y train limpio es histórico.
- Mínimos de pérdida F y pérdidas del checkpoint F1 se registran por separado con sus épocas en `perdidas_f.csv`. Nunca se sustituyen entre sí ni entre familias.
- El seleccionado sigue siendo A4-CNN-AdamW/LR, checkpoint 61/73: F1 0.5744742346, exactitud 64.43965517%, 2,392 aciertos de 3,712. Las 104 métricas por clase y matrices conservan esa única procedencia.
- Notebook y generador incluyen antecedentes A3/DM, reutilización F, comparaciones A4 y comprobación de procedencia sin entrenamiento. El ZIP incorpora las fuentes A3 intactas y los cuatro alias F necesarios para ejecutar la comprobación sin depender del checkout original.

## Verificación realizada

- 18 pruebas ligeras `unittest`: 6 de evidencia, 3 de regularización, 2 del reporte, 3 de alineación y 4 de entrega. Incluyen rechazo de métricas mezcladas, pérdida mínima frente a checkpoint, alteraciones/adiciones A3, ZIP antiguo y hashes de cada fuente del ZIP.
- `python -m scripts.check_avance3_avance4`: 13 clasificadores históricos A3, ocho archivos alias idénticos, cuatro definiciones de pérdida y todos los SHA-256 del A3 intacto. También se comprueba por `git diff` frente a la base que A3, reporte final y evidencia original A4 permanecen intactos.
- Notebook final ejecutado en modo evidencia: 11 celdas de código, sin errores, seis visualizaciones embebidas. El mismo notebook se ejecuta también desde una extracción aislada del ZIP, sin acceso a las entradas del checkout original.
- PDF compilado con Tectonic 0.17.0/XeTeX/BibTeX. Las 13 páginas se renderizaron con Poppler y se revisaron visualmente: portada, tablas, curvas, referencias, anexo de 104 clases y matriz. Sin `??`, texto fuera de página ni tablas desbordadas. Queda una advertencia de caja vacía de la clase y advertencias de anclas duplicadas; los destinos de las ocho tablas y siete figuras se cotejaron con las páginas reales.
- PDF y notebook canónicos, copias de entrega y entradas ZIP idénticos por contenido/SHA-256. Manifiesto de fuentes actualizado y ZIP íntegro sin checkpoints/datasets/residuos de construcción. El ZIP sí incluye la referencia documental A3 congelada.
- Sintaxis Python y celdas de notebook, esquema nbformat y manifiestos comprobados. Las auditorías anteriores se conservan; la validación vigente está en `validacion_entrega.json` y `correcciones_etapa5/validacion_pdf.json`.

## Bloqueos y límites

`python -m pytest --collect-only -q` continúa bloqueado por `ModuleNotFoundError: tensorflow` en `tests/conftest.py`. No se declara aprobada la suite completa. No se entrenó, no se descargaron datos y no se ejecutó inferencia sobre imágenes. Faltan TFRecords/frameworks y los checkpoints F locales: pérdidas, train limpio y paridad de exportación siguen siendo evidencia histórica. No hay test independiente ni variabilidad entre semillas. El entorno aislado bloqueó las conexiones locales Jupyter; la ejecución de evidencia se completó con permiso de ejecución fuera de ese aislamiento. La compilación necesitó acceso a la caché de Tectonic fuera del aislamiento.

## Reproducción

Desde la raíz, instalar únicamente `reports/entregable_4/requirements-evidence.txt` para el análisis:

```powershell
python -m scripts.check_avance3_avance4
python -m scripts.build_avance4_report
python -m scripts.build_avance4_notebook
python -m scripts.execute_avance4_notebook --output .venv/alineacion-build
python -m unittest tests.test_avance4_evidence tests.test_avance4_regularization tests.test_avance4_report tests.test_avance3_avance4_alignment tests.test_avance4_delivery -v
```

Tras recompilar desde `reports/entregable_4/` con `tectonic equipo_proyecto_avance4.tex`, renderizar y revisar las páginas; actualizar la validación de PDF con `scripts.validate_avance4_pdf` y las 13 páginas revisadas. Después ejecutar `python -m scripts.package_avance4` y las pruebas de entrega. Una regeneración del notebook exige volver a ejecutarlo antes de empaquetar. Las pruebas de entrega se ejecutan después de actualizar las copias/ZIP. No ejecutar el notebook A3 para esta revisión, pues regenera sus propias fuentes.

Entregas actualizadas: `entregables/avance4/equipo_proyecto_avance4.{pdf,ipynb,zip}`. Commits separados de trazabilidad, narrativa, notebook/generadores y validación final. PR hacia `main`, sin fusionarlo; enlace en el cierre de la conversación.
