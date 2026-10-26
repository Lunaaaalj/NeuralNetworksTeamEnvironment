"""Generate A4 numeric tables from historical evidence; never rewrite academic prose.
python -m scripts.build_avance4_report
"""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix
from scripts.analyze_mlp_avance4 import SPECS, validate_saved_predictions
from reports.entregable_4.correcciones_etapa3.analizar_regularizacion import load_evidence
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/entregable_4'
HISTORICAL=REPORT/'avance4_2026-10-21'
SHORT={'mlp_referencia':'MLP','cnn_original':'CNN original','cnn_dos_capas_extra':'CNN ampliada','cnn_regularizada':'CNN AdamW/LR'}

def escape(value):
    return str(value).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')

def table(path,caption,label,columns,header,rows,note=''):
    lines=[r'\begin{table*}[htbp]',r'\centering',r'\small',f'\\caption{{{caption}}}',f'\\label{{{label}}}',
           f'\\begin{{tabular}}{{{columns}}}',r'\toprule',header+r' \\',r'\midrule']
    lines += [row+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}']
    if note: lines += [r'\par\smallskip',r'\parbox{0.97\textwidth}{\footnotesize '+note+'}']
    lines += [r'\end{table*}']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def build(output=HISTORICAL):
    if Path(output).resolve()!=HISTORICAL.resolve():
        raise ValueError('Use audited historical A4 directory; training outputs are separate')
    tables,derived=REPORT/'tablas',REPORT/'correcciones_etapa4'
    tables.mkdir(exist_ok=True);derived.mkdir(exist_ok=True)
    sources,records,details={},[],{}
    def source(path):
        data=path.read_bytes()
        sources[path.relative_to(ROOT).as_posix()]={'sha256':hashlib.sha256(data).hexdigest(),
              'text_lf_sha256':hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()}
        return path
    names=pd.read_csv(source(ROOT/'reports/final/experimentos/conteo_por_clase.csv'))[['label','name']]
    for identifier,_,family,variant,run,_ in SPECS[:4]:
        d=ROOT/family/'runs'/variant/run
        s=json.loads(source(d/'summary.json').read_text(encoding='utf-8'))
        e=json.loads(source(d/'eval_report.json').read_text(encoding='utf-8'))
        p=pd.read_csv(source(d/'evaluation/validation_predictions.csv'))
        cm=pd.read_csv(source(d/'evaluation/confusion_matrix.csv'),index_col=0).to_numpy()
        h=pd.read_csv(source(d/'history.csv'))
        m=validate_saved_predictions(p,cm,e,s,h)
        classes=m.pop('per_class').merge(names,on='label',validate='one_to_one')
        records.append({'experiment_id':identifier,'variant':variant,'parameters':s['parameters'],
            'best_epoch':s['best_epoch'],'epochs_run':s['epochs_run'],**m,'val_loss_saved':e['val_loss']})
        details[variant]=(classes,p,h)
    results=pd.DataFrame(records)
    results.to_csv(derived/'resultados_a4_verificados.csv',index=False)
    chosen=results.loc[results.f1_macro.idxmax()]
    assert chosen.variant=='cnn_regularizada','Selection changed; revise interpretation explicitly'
    classes,p,h=details[chosen.variant]
    classes.to_csv(derived/'metricas_seleccionado.csv',index=False)
    cm=confusion_matrix(p.true_label,p.predicted_label,labels=range(104))
    pd.DataFrame(cm,index=range(104),columns=range(104)).to_csv(derived/'matriz_conteos.csv')
    pd.DataFrame(cm/cm.sum(1,keepdims=True),index=range(104),columns=range(104)).to_csv(derived/'matriz_normalizada.csv')
    errors=sorted([(int(cm[i,j]),i,j) for i in range(104) for j in range(104) if i!=j and cm[i,j]>0],key=lambda x:(-x[0],x[1],x[2]))
    name_map=names.set_index('label')['name'].to_dict()
    pairs=[{'true_label':i,'predicted_label':j,'true':name_map[i],'predicted':name_map[j],'count':n} for n,i,j in errors[:10]]
    selected={'experiment_id':chosen.experiment_id,'selection':'max macro F1 among four historical A4 runs',
        'checkpoint_epoch':int(chosen.best_epoch),'epochs_run':int(chosen.epochs_run),'samples':len(p),'correct':int(cm.trace()),
        'accuracy':float(chosen.accuracy),'f1_macro':float(chosen.f1_macro),'precision_macro':float(classes.precision.mean()),
        'recall_macro':float(classes.recall.mean()),'val_loss_saved':float(chosen.val_loss_saved),'confusions':pairs,
        'new_training':False,'new_image_inference':False}
    (derived/'evaluacion_seleccionado.json').write_text(json.dumps(selected,indent=2,ensure_ascii=False),encoding='utf-8')
    table(tables/'modelos_a4.tex','Familia A4: validación histórica recalculada desde predicciones (3,712 imágenes).',
        'tab:modelos-a4','lrrrrr','Modelo & Parámetros & Época/total & Exact. (\\%) & F1 macro & Pérdida',
        [f'{SHORT[r.variant]} & {r.parameters:,} & {r.best_epoch}/{r.epochs_run} & {100*r.accuracy:.2f} & {r.f1_macro:.4f} & {r.val_loss_saved:.4f}' for r in results.itertuples()],
        'Época seleccionada por F1 macro / épocas completas. PyTorch/DirectML, seed 0, entrada 112 por 112. La pérdida procede de evaluación guardada; no se repitió inferencia.')
    _,reg,_,reg_sources=load_evidence();sources.update(reg_sources)
    for context,filename,caption in [('selected_checkpoint','regularizacion_checkpoints','Familia F: mejores checkpoints por F1 macro, con y sin intervención conjunta.'),
                                     ('common_epoch_34','regularizacion_epoca34','Familia F: comparación complementaria en la época 34 de ambas corridas.')]:
        rows=reg[reg.comparison==context]
        table(tables/(filename+'.tex'),caption,'tab:'+filename,'lrrrrrrrr',
            'Variante & Época & $F_t$ & $F_v$ & $E_t$(\\%) & $E_v$(\\%) & $L_t$ & $L_v$ & $\\Delta F$',
            [f"{'Con' if r.experiment_id=='F/referencia' else 'Sin'} & {r.epoch} & {r.train_f1_macro:.4f} & {r.val_f1_macro:.4f} & {100*r.train_eval_accuracy:.2f} & {100*r.val_accuracy:.2f} & {r.train_eval_loss:.4f} & {r.val_loss:.4f} & {r.gap_f1:.4f}" for r in rows.itertuples()],
            '$t$: 3,712 imágenes fijas de train, sin aumentos y en inferencia; $v$: 3,712 de validación. $F$: F1 macro; $E$: exactitud; $L$: entropía cruzada; $\\Delta F=F_t-F_v$. TensorFlow/Metal según documentación histórica; train y pérdidas solo cotejados con históricos.')
    top=classes.sort_values(['f1','label'],ascending=[False,True]).head(4)
    bottom=classes.sort_values(['f1','label']).head(4)
    table(tables/'clases_resumen.tex','Modelo A4 seleccionado: cuatro clases con mayor y cuatro con menor F1.',
        'tab:clases-resumen','rlrrrr','ID & Clase & Precisión & Recall & F1 & Soporte',
        [f'{r.label} & {escape(r.name)} & {r.precision:.4f} & {r.recall:.4f} & {r.f1:.4f} & {r.support}' for r in pd.concat([top,bottom]).itertuples()],
        'Empates por ID. Las 104 clases se incluyen en el anexo y CSV.')
    table(tables/'confusiones.tex','Principales confusiones del checkpoint A4 seleccionado.','tab:confusiones','llr',
        'Clase real & Predicción & Imágenes',[f"{escape(r['true'])} & {escape(r['predicted'])} & {r['count']}" for r in pairs[:6]])
    lines=[r'\small',r'\begin{longtable}{rlrrrr}',
        r'\caption{Métricas de las 104 clases. A4-CNN-AdamW/LR, checkpoint 61.}\label{tab:clases-completas}\\',
        r'\toprule ID & Clase & Precisión & Recall & F1 & Soporte \\ \midrule',r'\endfirsthead',
        r'\multicolumn{6}{c}{Tabla de clases (continuación)}\\',
        r'\toprule ID & Clase & Precisión & Recall & F1 & Soporte \\ \midrule',r'\endhead',r'\bottomrule',r'\endfoot']
    lines += [f'{r.label} & {escape(r.name)} & {r.precision:.4f} & {r.recall:.4f} & {r.f1:.4f} & {r.support} '+r'\\' for r in classes.itertuples()]
    lines += [r'\end{longtable}']
    (tables/'clases_completas.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (derived/'fuentes.json').write_text(json.dumps({'sources':sources,'selection':selected,
        'method':'Derived from historical predictions and histories; originals unchanged'},indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(selected,indent=2,ensure_ascii=False))
    return selected

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HISTORICAL)
    build(parser.parse_args().output)
