"""Validate and package the PDF, executed notebook, figures and CSV evidence."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.config import ROOT


def package(output):
    output = Path(output).resolve()
    table = pd.read_csv(output / "resultados.csv")
    assert set(table.experiment) == {"mlp_referencia", "cnn_original", "cnn_dos_capas_extra", "cnn_regularizada"}
    for row in table.itertuples():
        directory = next((output / "runs" / row.experiment).glob("*/summary.json")).parent
        pred = pd.read_csv(directory / "evaluation/validation_predictions.csv")
        matrix = pd.read_csv(directory / "evaluation/confusion_matrix.csv", index_col=0).to_numpy()
        np.testing.assert_allclose(accuracy_score(pred.true_label, pred.predicted_label), row.val_accuracy, atol=1e-9)
        np.testing.assert_allclose(f1_score(pred.true_label, pred.predicted_label, average="macro"),
                                   row.val_f1_macro, atol=0.000051)
        assert matrix.shape == (104, 104)
        assert matrix.sum() == len(pred) == 3712
        assert matrix.trace() == pred.correct.sum()
        assert row.train_images == 12753
        evaluation = json.loads((directory / "eval_report.json").read_text(encoding="utf-8"))
        assert evaluation["keras_export_verified"]
        for name in ("curvas_entrenamiento_validacion", "confusion_matrix", "confusion_matrix_normalized", "predicciones_ejemplo"):
            for extension in ("png", "pdf"):
                assert (output / "figures" / row.experiment / f"{name}.{extension}").is_file()
    notebook = ROOT / "notebooks/equipo_proyecto_avance4.ipynb"
    nb = nbformat.read(notebook, as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is not None, "Execute all notebook cells before packaging"
            assert not any(item.output_type == "error" for item in cell.outputs)
    pdf = ROOT / "reports/equipo_proyecto_avance4.pdf"
    assert pdf.is_file() and pdf.stat().st_size > 20000
    delivery = ROOT / "entregables/avance4"
    delivery.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, delivery / pdf.name)
    shutil.copy2(notebook, delivery / notebook.name)
    instructions = """ENTREGA AVANCE 4

Subir a Canvas: equipo_proyecto_avance4.pdf y equipo_proyecto_avance4.ipynb.
El ZIP acompaña las figuras y la evidencia; no contiene los datos originales ni checkpoints grandes.

Figuras: arquitecturas, curvas por modelo, comparación, matrices de confusión y ejemplos de predicciones.
Métricas: resultados.csv, históricos por época, eval_report.json y predicciones individuales.
Los modelos .pt y .keras se conservan en reports/avance4_2026-10-21/models/.

La MLP está identificada como referencia reconstruida: no se encontró el artefacto exacto del Avance 3.
Las métricas corresponden a una semilla y a validación, no a prueba independiente.
Se atendió la observación compartida sobre representaciones; no se recibió el texto completo de la retroalimentación.

Reproducir: ejecutar las celdas del notebook; activar RUN_TRAINING para nuevas ejecuciones.
Construir PDF desde reports/: tectonic equipo_proyecto_avance4.tex.
El notebook usa los módulos de este repositorio y sus dependencias (requirements-directml.txt).
"""
    (delivery / "LEEME.txt").write_text(instructions, encoding="utf-8")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["teacher_feedback"] = "User reports missing model representations; full written feedback not provided"
    manifest["fully_connected_avance3"] = "Reconstructed reference; original Avance 3 artifact not identified"
    source_files = list((ROOT / "src").glob("*.py")) + [ROOT / "scripts" / name for name in
                    ("run_avance4.py", "prepare_avance4.py", "build_avance4_report.py", "package_avance4.py",
                     "execute_avance4_notebook.py")]
    manifest["packaged_source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in source_files}
    manifest["validated"] = {"runs": 4, "samples_each": 3712, "confusion_shape": [104, 104],
                              "notebook_executed_without_errors": True, "checkpoint_export_parity": True}
    manifest["package_versions"] = {package: version(package) for package in
                                    ("tensorflow", "torch", "torch-directml", "keras", "numpy",
                                     "scikit-learn", "matplotlib", "nbformat", "nbclient")}
    data_root = ROOT / "data/raw/tpu-getting-started/tfrecords-jpeg-224x224"
    manifest["dataset_sha256"] = {}
    for split in ("train", "val"):
        for path in sorted((data_root / split).glob("*.tfrec")):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest["dataset_sha256"][path.relative_to(ROOT).as_posix()] = digest
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    archive = delivery / "equipo_proyecto_avance4.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        for file in (delivery / pdf.name, delivery / notebook.name, delivery / "LEEME.txt"):
            bundle.write(file, file.name)
        for base in (output / "figures", output / "runs", output / "diagnosticos"):
            for file in base.rglob("*"):
                if file.is_file() and file.suffix.lower() in {".png", ".pdf", ".csv", ".json", ".yaml"}:
                    bundle.write(file, file.relative_to(ROOT).as_posix())
        for file in (list(output.glob("*.yaml")) + list(output.glob("*.json")) + list(output.glob("*.csv"))
                     + [output / "resumen_presentacion.txt"]):
            bundle.write(file, file.relative_to(ROOT).as_posix())
        for directory in (ROOT / "src", ROOT / "scripts", ROOT / "configs"):
            for file in directory.glob("*"):
                if file.is_file() and file.suffix in {".py", ".yaml"}:
                    bundle.write(file, file.relative_to(ROOT).as_posix())
        for filename in ("requirements.txt", "requirements-directml.txt"):
            bundle.write(ROOT / filename, filename)
        for file in (ROOT / "reports").rglob("*.tex"):
            if output not in file.parents:
                bundle.write(file, file.relative_to(ROOT).as_posix())
        bundle.write(ROOT / "reports/references.bib", "reports/references.bib")
        for name in ("cnn_diagram.png", "ejemplos_flores.png"):
            file = ROOT / "reports/figures" / name
            bundle.write(file, "reports/figures/" + file.name)
        bundle.write(output / "figures/dm_historico/PROCEDENCIA.txt",
                     (output / "figures/dm_historico/PROCEDENCIA.txt").relative_to(ROOT).as_posix())
    print(json.dumps({"delivery": str(delivery), "archive": str(archive),
                      "validated_runs": len(table)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    package(parser.parse_args().output)
