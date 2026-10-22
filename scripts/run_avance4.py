"""Rerun the CNN experiments in an isolated, dated Avance 4 directory."""

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from src.config import ROOT, load_config


def prepare_configs(destination):
    destination.mkdir(parents=True, exist_ok=True)
    specs = [
        ("cnn_original", "configs/directml_fast.yaml"),
        ("cnn_dos_capas_extra", "configs/directml_deep.yaml"),
        ("cnn_regularizada", "configs/directml_regularized.yaml"),
    ]
    configs = []
    for label, original in specs:
        cfg = deepcopy(load_config(ROOT / original))
        for key, folder in (("runs", "runs"), ("models", "models"), ("figures", "figures")):
            cfg["paths"][key] = destination / folder / label
        cfg["paths"]["results"] = destination / "resultados.csv"
        cfg.setdefault("wandb", {})["enabled"] = False
        serializable = deepcopy(cfg)
        serializable["paths"] = {key: str(value) for key, value in cfg["paths"].items()}
        path = destination / f"{label}.yaml"
        if not path.exists():
            path.write_text(yaml.safe_dump(serializable, allow_unicode=True, sort_keys=False), encoding="utf-8")
        configs.append((label, path))
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="*", help="Select experiment labels; completed runs are preserved.")
    args = parser.parse_args()
    destination = args.output.resolve()
    configs = prepare_configs(destination)
    manifest_path = destination / "manifest.json"
    if not manifest_path.exists():
        source_files = sorted((ROOT / "src").glob("*.py"))
        manifest = {
            "created_at": datetime.now().astimezone().isoformat(),
            "purpose": "Avance 4: reruns with seed 0, same train/validation split and resolution",
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in source_files},
            "fully_connected_avance3": "not_found; any new MLP is a reconstructed reference",
            "teacher_feedback": "not_provided",
            "selection_metric": "validation macro F1; no labeled independent test set",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    # Separate processes release the TensorFlow RAM cache and DirectML allocations between runs.
    import sys
    for label, config in configs:
        if args.only and label not in args.only:
            continue
        summaries = list((destination / "runs" / label).glob("*/summary.json"))
        if summaries:
            print(f"Preserving completed experiment: {label}", flush=True)
            continue
        print(f"\n=== {label} ===", flush=True)
        subprocess.run([sys.executable, "-u", "-m", "src.train", "--config", str(config),
                        "--data", "full", "--seed", "0"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
