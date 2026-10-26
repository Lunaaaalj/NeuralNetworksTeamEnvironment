# Cierre de correcciones del Avance 4

Rama: `fix/avance4-retroalimentacion`. Fecha: 26 de octubre de 2026.

## Estado por etapa

| Etapa | Estado | Evidencia / commit |
|---|---|---|
| 0–1 | Completadas previamente | Auditoría `e414f8f`, plan `dc2543f` |
| 2 | Diagnóstico histórico completado, sin entrenamiento nuevo | `749ddd0`, `e83625b`; curvas, predicciones y pesos MLP |
| 3 | Completada | `5da0023`; par F con/sin regularización, checkpoints y época común 34 |
| 4 | Completada | `f9c7b6a`; ocho secciones académicas, tablas y anexo completo |
| 5 | Entregables generados y verificados | Notebook ejecutado, PDF compilado/revisado, copias y ZIP sincronizados; commit que contiene este cierre |

## Métricas utilizadas

| Familia / modelo | F1 macro val | Exactitud val | Checkpoint / épocas |
|---|---:|---:|---:|
| A4 MLP | 0.0280 | 15.52% | 77 / 87 |
| A4 CNN original | 0.5740 | 63.25% | 53 / 58 |
| A4 CNN ampliada | 0.5502 | 62.23% | 46 / 51 |
| A4 CNN AdamW/LR, seleccionada | 0.5745 | 64.44% | 61 / 73 |
| F con intervención conjunta | 0.5784 | 64.06% | 91 / 103 |
| F sin esas intervenciones | 0.5332 | 59.64% | 34 / 46 |

Solo el grupo A4 interviene en la elección del modelo A4. F es evidencia complementaria para regularización. En F, la brecha F1 limpia es 0.2354 con regularización y 0.3435 sin ella. El efecto observado al comparar mejores checkpoints es +0.045155 F1 y +4.4181 puntos porcentuales de exactitud; la brecha se reduce 0.108023. En época 34, sin regularización tiene mayor F1/exactitud, mientras con regularización tiene menor pérdida y brecha. Se conserva este matiz y no se atribuye la mejora a una sola técnica.

## Criterios de la rúbrica atendidos

- Identificación correcta del Avance 4, ocho secciones, algoritmos desarrollados y enlace al repositorio/notebook.
- Tabla con/sin regularización, curvas de ambos splits y brechas limpias; comparación principal de checkpoints y complemento en época 34 sin interpolación.
- MLP realmente densa, parámetros, comparación explícita con CNN, curvas y explicación de bajo ajuste sin afirmar incapacidad universal.
- CNN ampliada con F1 y análisis de pérdida/sobreajuste, diferenciando checkpoint y épocas ejecutadas.
- Modelo seleccionado con precisión, recall, F1 y soporte de 104 clases, matriz y principales confusiones; anexo completo.
- Notebook organizado con análisis histórico predeterminado, entrenamiento opcional separado y limitaciones explícitas.

## Pruebas aprobadas

- 14 pruebas ligeras mediante `unittest`: consistencia de métricas y matrices, 104 clases, selección, épocas, rechazo de mezclas y ZIP antiguo, sincronización de entregables.
- Notebook final: diez celdas de código ejecutadas sin errores y seis visualizaciones guardadas; gráficas revisadas.
- Notebook del ZIP: ejecución completa desde una extracción aislada en carpeta ignorada, sin depender de datos/frameworks de entrenamiento.
- Sintaxis Python de src/scripts/tests y código de notebook; esquemas nbformat; CSV/JSON de evidencia y derivados.
- Metadatos históricos de cuatro PT inspeccionados con `pickletools`, sin cargar modelos: épocas 77, 53, 46 y 61 coherentes con sus resúmenes.
- PDF: Tectonic 0.17.0 ejecutó TeX/BibTeX con éxito. Se renderizaron e inspeccionaron las 12 páginas, sin `??` ni texto fuera de página; destinos internos de tablas y figuras comprobados. Las advertencias de la clase LaTeX constan en `validacion_pdf.json` y no impidieron la compilación ni referencias correctas.
- Copias de PDF/ipynb idénticas por SHA-256 y contenido idéntico dentro del ZIP. ZIP íntegro, sin datasets/checkpoints ni residuos temporales.
- Git: entregas 1/3, reporte final y evidencia original A4 sin modificaciones. `main` intacta.

## Pruebas bloqueadas y limitaciones

`python -m pytest --collect-only -q` falla al importar TensorFlow en `tests/conftest.py`. No se declara aprobada la suite completa. No hay TFRecords ni frameworks de entrenamiento en el Python activo. No se ejecutaron nuevos entrenamientos, inferencia de imágenes, pérdidas ni train limpio. F1/exactitud/clases/matrices se recalcularon desde predicciones; pérdidas y métricas limpias de train se verificaron documentalmente. La MLP histórica queda con limitaciones de ajuste. No se demuestra contribución individual de regularizadores, significancia estadística ni rendimiento de test independiente.

## Archivos finales y reproducción

Archivos para entregar: `entregables/avance4/equipo_proyecto_avance4.pdf` y `entregables/avance4/equipo_proyecto_avance4.ipynb`. El ZIP de la misma carpeta está actualizado. Fuentes canónicas: `reports/entregable_4/` y `notebooks/equipo_proyecto_avance4.ipynb`.

`requirements-evidence.txt` registra versiones del entorno ligero ejecutado; los requisitos de entrenamiento permanecen separados. `scripts/build_avance4_report.py` regenera números sin reescribir prosa; `scripts/build_avance4_notebook.py` reconstruye el notebook (después debe ejecutarse). `scripts/package_avance4.py` valida y sincroniza copias sin tocar manifiestos históricos. `scripts/validate_avance4_pdf.py` verifica el PDF y exige indicar las páginas revisadas visualmente.

Los manifiestos `validacion_pdf.json`, `validacion_entrega.json` y `manifiesto_entrega.json` registran comprobaciones y hashes. La referencia del PR y el hash del commit de etapa 5 se proporcionan en el cierre de la conversación y en el historial Git; no se incluye un hash autorreferente dentro del propio commit. PR hacia `main`, sin fusión automática.
