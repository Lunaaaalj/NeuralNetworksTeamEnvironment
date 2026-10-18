"""Descarga los TFRecords de 224 px de Kaggle desde Windows, macOS o Linux."""

import argparse
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "tpu-getting-started"


def download(dest):
    from kaggle.api.kaggle_api_extended import KaggleApi

    dest = dest.resolve()
    api = KaggleApi()
    api.authenticate()
    files, token = [], None
    while True:
        page = api.competition_list_files(COMPETITION, page_token=token, page_size=200)
        files.extend(page.files)
        token = page.next_page_token
        if not token:
            break

    wanted = [f for f in files if "224x224" in f.name or f.name == "sample_submission.csv"]
    if not wanted:
        raise RuntimeError("Kaggle no devolvió los archivos de 224x224 de la competencia.")
    for index, item in enumerate(wanted, 1):
        target = (dest / item.name).resolve()
        if not target.is_relative_to(dest):
            raise ValueError(f"Ruta de descarga fuera del destino: {item.name}")
        if target.exists() and target.stat().st_size == item.total_bytes:
            print(f"[{index}/{len(wanted)}] Ya descargado: {item.name}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(wanted)}] Descargando: {item.name}", flush=True)
        api.competition_download_file(COMPETITION, item.name, path=str(target.parent), quiet=False)
        archive_path = target.with_name(target.name + ".zip")
        if archive_path.exists():
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    if not (target.parent / member.filename).resolve().is_relative_to(target.parent):
                        raise ValueError(f"Ruta ZIP fuera del destino: {member.filename}")
                archive.extractall(target.parent)
            archive_path.unlink()

    tfrecords = dest / "tfrecords-jpeg-224x224"
    for split in ("train", "val", "test"):
        if not any((tfrecords / split).glob("*.tfrec")):
            raise RuntimeError(f"No se encontraron TFRecords en {tfrecords / split}")
    print(f"Listo: {tfrecords}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=ROOT / "data/raw/tpu-getting-started")
    args = parser.parse_args()
    download(args.dest)


if __name__ == "__main__":
    main()
