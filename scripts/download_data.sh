#!/usr/bin/env bash
# Descarga solo los TFRecords 224x224 de "Petals to the Metal" (0.75 GB de los 5.15 GB del zip
# completo) a data/raw/tpu-getting-started/tfrecords-jpeg-224x224/{train,val,test}.
# Requiere: pip install kaggle, un token en ~/.kaggle (access_token o kaggle.json) y haber
# aceptado las reglas en https://www.kaggle.com/competitions/tpu-getting-started
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/raw/tpu-getting-started"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python

"$PYTHON" - "$DEST" <<'EOF'
import sys
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMP, dest = "tpu-getting-started", Path(sys.argv[1])
api = KaggleApi()
api.authenticate()

files, token = [], None
while True:
    page = api.competition_list_files(COMP, page_token=token, page_size=200)
    files += page.files
    token = page.next_page_token
    if not token:
        break

wanted = [f for f in files if "224x224" in f.name or f.name == "sample_submission.csv"]
for i, f in enumerate(wanted, 1):
    target = dest / f.name
    if target.exists() and target.stat().st_size == f.total_bytes:
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{i}/{len(wanted)}] {f.name}", flush=True)
    api.competition_download_file(COMP, f.name, path=str(target.parent), quiet=True)

# La API entrega algunos archivos como <nombre>.zip: se extraen en su lugar.
for z in dest.rglob("*.zip"):
    with zipfile.ZipFile(z) as zf:
        zf.extractall(z.parent)
    z.unlink()
print(f"Listo: {dest / 'tfrecords-jpeg-224x224'}")
EOF
