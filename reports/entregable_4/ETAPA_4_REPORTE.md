# Etapa 4: integración del reporte

Se actualizaron las ocho secciones académicas del A4 y se añadió un anexo de 104 clases. Se conservan las fuentes y resultados originales de A4 y del reporte final.

Cambios: algoritmos desarrollados, metodología a 112×112, familias A4/F separadas, diagnóstico de MLP y CNN ampliada, tablas de regularización por checkpoint y época 34, curvas, selección exclusivamente entre los cuatro A4 y métricas por clase derivadas de sus predicciones. No se introdujeron entrenamientos nuevos ni se atribuyeron efectos individuales a los regularizadores.

El generador `scripts/build_avance4_report.py` ahora produce solo tablas numéricas y derivados verificados; no reescribe las secciones ni el notebook. Así una regeneración no restaura la narrativa anterior. Los archivos nuevos están en `tablas/` y `correcciones_etapa4/`, con rutas/hashes de fuentes. El seleccionado sigue siendo A4_CNN_AdamW: checkpoint 61/73, 2,392 aciertos/3,712, F1 0.5744742346, exactitud 0.6443965517, precisión macro 0.6214684170 y recall macro 0.5634652237.

Verificación: tablas generadas desde predicciones/históricos, referencias LaTeX y figuras existentes, dos pruebas unitarias de reconciliación por clase y selección A4. Compilación preliminar Tectonic exitosa en carpeta ignorada, sin referencias `??`; el PDF final, notebook ejecutado, revisión visual completa y sincronización del ZIP corresponden a etapa 5.

Límites: sin TFRecords/frameworks no se repitieron inferencia de imágenes, pérdidas ni entrenamiento limpio. Estos campos se presentan como evidencia histórica. No se modificaron entregas anteriores ni el reporte final.
