"""Package current verified A4 delivery; original experimental manifests stay read-only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile,ZIP_DEFLATED
import nbformat
import pandas as pd
from scripts.build_avance4_report import HISTORICAL,ROOT,REPORT


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_archive(archive,pdf,notebook):
    with ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert bundle.read(pdf.name)==pdf.read_bytes(),'ZIP PDF stale'
        assert bundle.read(notebook.name)==notebook.read_bytes(),'ZIP notebook stale'
        assert bundle.read('notebooks/'+notebook.name)==notebook.read_bytes()
        names=bundle.namelist()
        assert len(names)==len(set(names))
        assert not any(any(word in n for word in ['pytest-temp','pytest-atomic-temp','latex-build','.venv','__pycache__']) for n in names)
        assert not any(n.endswith(('.pt','.keras','.tfrec','.tmp')) for n in names)
    return len(names)


def package(output=HISTORICAL):
    if Path(output).resolve()!=HISTORICAL.resolve():raise ValueError('Use audited historical A4 evidence')
    notebook=ROOT/'notebooks/equipo_proyecto_avance4.ipynb'
    pdf=REPORT/'equipo_proyecto_avance4.pdf'
    nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb)
    code=[cell for cell in nb.cells if cell.cell_type=='code']
    assert code and all(cell.execution_count is not None for cell in code)
    assert not any(o.output_type=='error' for cell in code for o in cell.outputs)
    assert pdf.is_file() and pdf.stat().st_size>20000
    review=REPORT/'correcciones_etapa5/validacion_pdf.json'
    checked=json.loads(review.read_text(encoding='utf-8'))
    assert checked['pdf_sha256']==digest(pdf) and checked['pages_visually_reviewed']==list(range(1,checked['pages']+1))
    assert checked['no_missing_references'] and checked['no_text_outside_page']
    classes=pd.read_csv(REPORT/'correcciones_etapa4/metricas_seleccionado.csv')
    selected=json.loads((REPORT/'correcciones_etapa4/evaluacion_seleccionado.json').read_text(encoding='utf-8'))
    assert len(classes)==104 and classes.support.sum()==3712
    assert abs(classes.f1.mean()-selected['f1_macro'])<1e-14
    delivery=ROOT/'entregables/avance4';delivery.mkdir(exist_ok=True,parents=True)
    shutil.copy2(pdf,delivery/pdf.name);shutil.copy2(notebook,delivery/notebook.name)
    instructions='''ENTREGA CORREGIDA AVANCE 4 — RED CONVOLUTIVA

Subir: equipo_proyecto_avance4.pdf y equipo_proyecto_avance4.ipynb.
El notebook está ejecutado en modo evidencia: no entrena por defecto ni necesita TFRecords/TensorFlow para mostrar resultados históricos. Requiere Python 3.12, numpy, pandas, scipy, scikit-learn, matplotlib, PyYAML y Jupyter para volver a ejecutarlo.

Familia A3: CNN inicial y reducción/DM, antecedentes del proyecto. Familia A4: cuatro corridas históricas DirectML; MLP reconstruida posteriormente y CNN compartida de 480,936 parámetros, con ejecuciones diferentes del A3. Familia F: hiperparámetros reutilizados en A3 corregido y reporte final, sin contar copias como repeticiones. Se profundiza regularización con checkpoints y época común 34. Pérdida mínima y pérdida del checkpoint F1 son cantidades distintas. No se mezclan métricas entre familias ni se usan datos DM en las cuatro corridas A4.

Reproducir desde la raíz del ZIP/repo:
python -m pip install -r reports/entregable_4/requirements-evidence.txt
python -m scripts.build_avance4_report
python -m scripts.check_avance3_avance4
python reports/entregable_4/correcciones_etapa3/analizar_regularizacion.py
python -m unittest tests.test_avance4_evidence tests.test_avance4_regularization tests.test_avance4_report tests.test_avance4_delivery -v
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/equipo_proyecto_avance4.ipynb

Compilar desde reports/entregable_4/: tectonic equipo_proyecto_avance4.tex
Empaquetar desde la raíz: python -m scripts.package_avance4
La validación visual del PDF debe renovarse si cambian sus fuentes; el empaquetado verifica el hash revisado.

El ZIP contiene las fuentes actuales, el notebook idéntico al entregado, curvas/tablas y CSV/JSON de evidencia necesarios para el análisis final; excluye datos, checkpoints y residuos de pruebas. El diagnóstico detallado de pesos de etapa 2 requiere los checkpoints del repositorio completo para repetirse; su lectura es opcional y no afecta al modo evidencia de este notebook.

No se hicieron nuevos entrenamientos ni inferencia sobre imágenes: faltaron TFRecords y frameworks. F1/exactitud/métricas por clase se recalcularon desde predicciones; pérdidas y train limpio se verificaron documentalmente. La suite pytest completa sigue requiriendo TensorFlow; no se declara aprobada. La MLP histórica se conserva con bajo ajuste. Una semilla y validación usada para selección limitan las conclusiones.
'''
    (delivery/'LEEME.txt').write_text(instructions,encoding='utf-8')
    files={}
    def include(path,arc=None):
        path=Path(path)
        assert path.is_file(),path
        files[arc or path.relative_to(ROOT).as_posix()]=path
    for p in [delivery/pdf.name,delivery/notebook.name,delivery/'LEEME.txt']:include(p,p.name)
    include(notebook)
    for folder in ['src','scripts','configs','tests']:
        for p in (ROOT/folder).glob('*'):
            if p.is_file() and p.suffix in {'.py','.yaml'}:include(p)
    for p in [ROOT/'requirements.txt',ROOT/'requirements-directml.txt',REPORT/'requirements-evidence.txt']:include(p)
    for folder in [REPORT/'secciones',REPORT/'tablas']:
        for p in folder.glob('*.tex'):include(p)
    for p in [REPORT/'main.tex',REPORT/'equipo_proyecto_avance4.tex',REPORT/'references.bib']:include(p)
    for stage in [2,3,4,5]:
        for p in (REPORT/f'correcciones_etapa{stage}').rglob('*'):
            if p.is_file() and p.name!='manifiesto_entrega.json' and p.suffix in {'.csv','.json','.md','.py','.png','.pdf'}:include(p)
    for p in [REPORT/'AUDITORIA_CORRECCIONES.md',REPORT/'PLAN_EXPERIMENTAL.md',REPORT/'ETAPA_2_MLP_CNN.md',REPORT/'ETAPA_4_REPORTE.md',REPORT/'ALINEACION_AVANCE3_AVANCE4.md']:include(p)
    for p in (REPORT/'alineacion_avance3').glob('*'):
        if p.is_file() and p.suffix in {'.md','.json','.csv'}:include(p)
    # A3 is a read-only reference; include its frozen files so provenance checks run offline.
    for p in (ROOT/'reports/entregable_3').rglob('*'):
        if p.is_file():include(p)
    for p in (HISTORICAL/'runs').rglob('*'):
        if p.is_file() and p.suffix in {'.csv','.json'}:include(p)
    for p in list(HISTORICAL.glob('*.yaml'))+[HISTORICAL/'manifest.json']:include(p)
    figure_names=['arquitecturas','comparacion_original_ampliada','cnn_regularizada/confusiones_principales','cnn_regularizada/confusion_matrix_normalized']
    for name in figure_names:
        for ext in ['png','pdf']:include(HISTORICAL/'figures'/f'{name}.{ext}')
    # Include F/mlp as provenance support for the stage-2 tables, separately from the regularization pair.
    final=ROOT/'reports/final/experimentos'
    include(final/'conteo_por_clase.csv')
    for variant in ['referencia','sin_regularizacion','mlp','lr_3e-4','dropout_0.3']:
        include(final/'configs'/f'{variant}.yaml')
        for p in (final/'runs'/variant).rglob('*'):
            if p.is_file() and p.suffix in {'.csv','.json'}:include(p)
    manifest={'purpose':'Current corrected A4 delivery; original evidence unchanged',
        'files_sha256':{name:digest(p) for name,p in sorted(files.items())},
        'notebook_code_cells_executed':len(code),'new_training':False,'new_image_inference':False,
        'limitations':['Loss and clean-train metrics checked against histories, not re-inferred',
                       'Full TensorFlow test suite unavailable','Single seed; no independent labeled test']}
    manifest_path=REPORT/'correcciones_etapa5/manifiesto_entrega.json'
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    include(manifest_path)
    archive=delivery/'equipo_proyecto_avance4.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as bundle:
        for name,p in sorted(files.items()):bundle.write(p,name)
    count=verify_archive(archive,pdf,notebook)
    assert digest(pdf)==digest(delivery/pdf.name) and digest(notebook)==digest(delivery/notebook.name)
    print(json.dumps({'archive_entries':count,'pdf_sha256':digest(pdf),'notebook_sha256':digest(notebook),
                      'delivery':str(delivery)},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=HISTORICAL)
    package(parser.parse_args().output)
