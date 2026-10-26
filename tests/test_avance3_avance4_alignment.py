"""Check actual shared evidence and distinguish loss selection; no training imports."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.check_avance3_avance4 import ROOT, OUT, digest, inspect, verify_frozen_a3


class AlignmentTest(unittest.TestCase):
    def test_aliases_are_same_evidence_and_a3_is_frozen(self):
        manifest = json.loads((OUT/'procedencia.json').read_text(encoding='utf-8'))
        verify_frozen_a3(manifest)
        actual = inspect()
        self.assertEqual(actual['aliases'], manifest['aliases'])
        self.assertEqual(actual['a3_classifiers_checked'], 13)
        self.assertEqual(actual['shared_cnn_parameters'], 480936)

    def test_minimum_loss_is_not_checkpoint_loss(self):
        rows = {r['experiment_id']:r for r in inspect()['loss_definitions']}
        for key, epochs in [('referencia',(91,54)),('sin_regularizacion',(34,25))]:
            r = rows['F/'+key]
            self.assertEqual((r['checkpoint_epoch'],r['minimum_loss_epoch']),epochs)
            self.assertLess(r['minimum_val_loss'],r['checkpoint_val_loss'])

    def test_frozen_manifest_rejects_modified_and_added_a3_files(self):
        with tempfile.TemporaryDirectory(prefix='.avance4-test-', dir=ROOT) as tmp:
            folder = Path(tmp)
            p = folder/'reference.txt'; p.write_text('approved')
            manifest = {'a3_files_sha256':{p.relative_to(ROOT).as_posix():digest(p)}}
            with patch('scripts.check_avance3_avance4.A3',folder):
                verify_frozen_a3(manifest)
                p.write_text('modified')
                with self.assertRaisesRegex(AssertionError,'A3 changed'):verify_frozen_a3(manifest)
                p.write_text('approved'); (folder/'extra.txt').write_text('added')
                with self.assertRaisesRegex(AssertionError,'A3 changed'):verify_frozen_a3(manifest)


if __name__ == '__main__':unittest.main()
