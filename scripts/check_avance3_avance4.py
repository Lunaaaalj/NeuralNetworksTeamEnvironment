"""Read-only provenance checks; --write emits A4 alignment evidence, never A3 files."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A3 = ROOT / 'reports/entregable_3'
OUT = ROOT / 'reports/entregable_4/alineacion_avance3'
VARIANTS = ['referencia', 'sin_regularizacion', 'lr_3e-4', 'dropout_0.3']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect():
    aliases, losses = [], []
    for variant in VARIANTS:
        original = A3 / 'hiperparametros' / variant
        final = next((ROOT / 'reports/final/experimentos/runs' / variant).glob('*/summary.json')).parent
        for name in ['summary.json', 'history.csv']:
            left, right = original / name, final / name
            assert left.read_bytes() == right.read_bytes(), f'Alias differs: {left}'
            aliases.append({'experiment_id': 'F/' + variant, 'a3': left.relative_to(ROOT).as_posix(),
                            'final': right.relative_to(ROOT).as_posix(), 'sha256': digest(left)})
        summary = json.loads((original / 'summary.json').read_text(encoding='utf-8'))
        history = pd.read_csv(original / 'history.csv')
        best = history.loc[history.epoch + 1 == summary['best_epoch']].iloc[0]
        minimum = history.loc[history.val_loss.idxmin()]
        assert abs(best.val_f1_macro - history.val_f1_macro.max()) < 1e-12
        losses.append({'experiment_id': 'F/' + variant, 'checkpoint_epoch': summary['best_epoch'],
                       'checkpoint_val_loss': float(best.val_loss), 'minimum_loss_epoch': int(minimum.epoch) + 1,
                       'minimum_val_loss': float(minimum.val_loss)})
    runs = list((A3 / 'runs').glob('*/summary.json'))
    assert len(runs) == 13
    for path in runs:
        summary = json.loads(path.read_text(encoding='utf-8'))
        history = pd.read_csv(path.with_name('history.csv'))
        assert len(history) == summary['epochs_run']
        best = history.loc[history.epoch + 1 == summary['best_epoch']].iloc[0]
        assert abs(best.val_f1_macro - summary['val_f1_macro']) <= 0.000051
        assert abs(best.val_f1_macro - history.val_f1_macro.max()) < 1e-12
    # Counts include biases: 896 + 18,496 + 73,856 + 295,168 + 65,792 + 26,728.
    parameters = sum((9 * i + 1) * o for i, o in [(3, 32), (32, 64), (64, 128), (128, 256)])
    parameters += (256 + 1) * 256 + (256 + 1) * 104
    assert parameters == 480936
    return {'families': {'A3': 'initial CNN and random/DM (13 classifier runs; two shared distillations)',
                         'A4': 'four DirectML comparisons; separate executions',
                         'F': 'hyperparameter runs reused in corrected A3; aliases are not replicas'},
            'a3_classifiers_checked': len(runs), 'shared_cnn_parameters': parameters,
            'aliases': aliases, 'loss_definitions': losses,
            'a3_files_sha256': {p.relative_to(ROOT).as_posix(): digest(p)
                                for p in sorted(A3.rglob('*')) if p.is_file()},
            'new_training': False, 'new_image_inference': False}


def verify_frozen_a3(manifest):
    actual = {p.relative_to(ROOT).as_posix(): digest(p) for p in A3.rglob('*') if p.is_file()}
    assert actual == manifest['a3_files_sha256'], 'A3 changed or files were added/removed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    result = inspect()
    if args.write:
        OUT.mkdir(exist_ok=True)
        (OUT / 'procedencia.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        pd.DataFrame(result['loss_definitions']).to_csv(OUT / 'perdidas_f.csv', index=False)
    else:
        verify_frozen_a3(json.loads((OUT / 'procedencia.json').read_text(encoding='utf-8')))
    print('A3 intact; 13 A3 classifiers, 8 identical alias files, 4 F loss definitions verified.')
