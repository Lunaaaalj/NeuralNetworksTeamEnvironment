"""Barrido de experimentos: completo vs. subconjunto aleatorio vs. destilado DM, varias semillas.

Destila los IPC que falten, entrena cada combinación y agrega una fila por corrida a
`paths.results` (reports/resultados.csv). Al final imprime media ± desviación por configuración.

Uso (desde la raíz del repo):
    python -m scripts.run_experiments --config configs/local.yaml
"""

import argparse
from pathlib import Path

import pandas as pd

from src.classes import CLASSES
from src.config import load_config, set_seed
from src.data import ClassCache, build_dataset, cache_path
from src.distill import distill, save_grid, save_synthetic, synthetic_path
from src.train import run


def ensure_distilled(cfg, ipc):
    """Destila con la configuración del YAML si el .npz para este IPC aún no existe."""
    img_size = cfg["data"]["img_size"]
    out = synthetic_path(cfg["paths"]["processed"], ipc, img_size)
    if out.exists():
        return out
    d = cfg["distill"]
    set_seed(0)
    cache = ClassCache(cache_path(cfg["paths"]["processed"], img_size))
    images, labels, _, seconds = distill(
        cache, ipc, d["iterations"], d["real_batch"], d["classes_per_step"], d["lr_img"],
        seed=0, log_every=d["log_every"],
        log_path=Path(cfg["paths"]["runs"]) / f"distill_ipc{ipc}_res{img_size}" / "loss.csv")
    save_synthetic(out, images, labels, {"ipc": ipc, "img_size": img_size, "iterations": d["iterations"],
                                         "seed": 0, "distill_seconds": round(seconds, 1)})
    save_grid(images, labels, Path(cfg["paths"]["figures"]) / f"synthetic_ipc{ipc}_res{img_size}.png", CLASSES)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/local.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp = cfg["experiments"]
    results_path = Path(cfg["paths"]["results"])
    results_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = [("full", None, s) for s in exp["seeds"]] if exp.get("include_full") else []
    jobs += [(d, ipc, s) for d in exp["data"] for ipc in exp["ipcs"] for s in exp["seeds"]]

    if "distilled" in exp["data"]:
        for ipc in exp["ipcs"]:
            ensure_distilled(cfg, ipc)

    # El dataset de val se construye una sola vez y se reutiliza en todas las corridas.
    val_ds = build_dataset(cfg["paths"]["tfrecords"], "val", cfg["data"]["img_size"], 128).cache()
    for data, ipc, seed in jobs:
        print(f"\n=== {data} ipc={ipc} seed={seed} ===")
        row = run(cfg, data, ipc, seed, val_ds=val_ds)
        pd.DataFrame([row]).to_csv(results_path, mode="a", header=not results_path.exists(), index=False)

    df = pd.read_csv(results_path)
    df["ipc"] = df["ipc"].fillna(0).astype(int)  # 0 = conjunto completo
    agg = df.groupby(["data", "ipc"]).agg(
        runs=("seed", "count"),
        f1_mean=("val_f1_macro", "mean"), f1_std=("val_f1_macro", "std"),
        acc_mean=("val_accuracy", "mean"), train_s=("train_seconds", "mean"),
    ).round(4)
    if "distill_seconds" in df:
        agg["distill_s"] = df.groupby(["data", "ipc"])["distill_seconds"].first()
    print("\n", agg.to_string())


if __name__ == "__main__":
    main()
