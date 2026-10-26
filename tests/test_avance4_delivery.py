"""Delivery synchronization and stale-archive rejection; no training dependencies."""
import tempfile
import hashlib
import json
from pathlib import Path
import unittest
from zipfile import ZipFile
import nbformat
from scripts.package_avance4 import ROOT, digest, verify_archive


class DeliveryTest(unittest.TestCase):
    def test_zip_manifest_matches_every_archived_source(self):
        with ZipFile(ROOT/'entregables/avance4/equipo_proyecto_avance4.zip') as bundle:
            manifest = json.loads(bundle.read('reports/entregable_4/correcciones_etapa5/manifiesto_entrega.json'))
            for name, expected in manifest['files_sha256'].items():
                self.assertEqual(hashlib.sha256(bundle.read(name)).hexdigest(), expected, name)
            self.assertIn('reports/entregable_4/ALINEACION_AVANCE3_AVANCE4.md', bundle.namelist())
            frozen = json.loads(bundle.read('reports/entregable_4/alineacion_avance3/procedencia.json'))
            for name, expected in frozen['a3_files_sha256'].items():
                self.assertEqual(hashlib.sha256(bundle.read(name)).hexdigest(), expected, name)

    def test_canonical_and_delivery_copies_are_identical(self):
        delivery = ROOT / "entregables/avance4"
        pdf = ROOT / "reports/entregable_4/equipo_proyecto_avance4.pdf"
        notebook = ROOT / "notebooks/equipo_proyecto_avance4.ipynb"
        self.assertEqual(digest(pdf), digest(delivery/pdf.name))
        self.assertEqual(digest(notebook), digest(delivery/notebook.name))
        self.assertGreater(verify_archive(delivery/"equipo_proyecto_avance4.zip", pdf, notebook), 50)

    def test_stale_notebook_in_archive_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix=".avance4-test-", dir=ROOT) as temp:
            root = Path(temp)
            self.assertTrue(root.resolve().is_relative_to(ROOT.resolve()))
            pdf, notebook = root/"report.pdf", root/"notebook.ipynb"
            pdf.write_bytes(b"current PDF")
            notebook.write_bytes(b"current notebook")
            archive = root/"stale.zip"
            with ZipFile(archive,"w") as bundle:
                bundle.writestr(pdf.name,pdf.read_bytes())
                bundle.writestr(notebook.name,b"old notebook")
                bundle.writestr("notebooks/"+notebook.name,b"old notebook")
            with self.assertRaisesRegex(AssertionError,"notebook stale"):
                verify_archive(archive,pdf,notebook)

    def test_evidence_notebook_has_executed_outputs_without_errors(self):
        notebook = nbformat.read(ROOT/"notebooks/equipo_proyecto_avance4.ipynb",as_version=4)
        nbformat.validate(notebook)
        cells = [c for c in notebook.cells if c.cell_type=="code"]
        self.assertEqual(len(cells),11)
        self.assertTrue(all(c.execution_count is not None for c in cells))
        self.assertFalse(any(o.output_type=="error" for c in cells for o in c.outputs))
        self.assertGreaterEqual(sum("image/png" in o.get("data",{}) for c in cells for o in c.outputs),5)


if __name__ == "__main__":
    unittest.main()
