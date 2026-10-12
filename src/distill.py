"""Destilación de datos por Distribution Matching (DM; Zhao & Bilen, WACV 2023).

Aprende `ipc` imágenes sintéticas por clase haciendo que el promedio de sus características
coincida con el de las imágenes reales de la misma clase, medido con redes de pesos aleatorios
que se vuelven a muestrear en cada paso:

    L = sum_c || mean f_theta(A(R_c)) - mean f_theta(A(S_c)) ||^2,   theta ~ P(theta) aleatorio

donde A es la augmentación siamesa DSA. Solo se optimizan los píxeles de S; la red nunca se
entrena, por eso DM es mucho más barato que los métodos de emparejar gradientes o trayectorias.

Uso:
    python -m src.distill --config configs/local.yaml --ipc 10
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.augment import dsa
from src.model import build_embedder


def synthetic_path(processed_dir, ipc, img_size):
    return Path(processed_dir) / f"synthetic_ipc{ipc}_res{img_size}.npz"


def load_synthetic(path):
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    return data["images"].astype(np.float32), data["labels"], meta


class DistributionMatcher:
    """Mantiene las imágenes sintéticas (C, ipc, H, W, 3) y aplica un paso de DM por grupo de clases."""

    def __init__(self, init_images, lr_img=0.01, beta1=0.9, beta2=0.999):
        self.syn = tf.Variable(init_images, dtype=tf.float32, name="synthetic")
        # Estado de Adam por píxel y contador de pasos por clase (cada clase avanza a su ritmo).
        self.m = tf.Variable(tf.zeros_like(init_images), name="adam_m")
        self.v = tf.Variable(tf.zeros_like(init_images), name="adam_v")
        self.t = tf.Variable(tf.zeros([init_images.shape[0]]), name="adam_t")
        self.lr_img, self.beta1, self.beta2 = lr_img, beta1, beta2
        self.embedder = build_embedder(input_shape=init_images.shape[2:])

    def _reinit_embedder(self, seed):
        """Nuevos pesos aleatorios (Glorot uniforme, como Keras por defecto) en cada paso."""
        for i, v in enumerate(self.embedder.trainable_variables):
            if len(v.shape) == 1:  # sesgos: se quedan en cero
                continue
            receptive = math.prod(v.shape[:-2])
            fan_in, fan_out = receptive * v.shape[-2], receptive * v.shape[-1]
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            v.assign(tf.random.stateless_uniform(
                v.shape, tf.random.experimental.stateless_fold_in(seed, i), -limit, limit))

    def _mean_features(self, batches, seeds):
        """batches: (g, n, H, W, 3). Aplica DSA por clase y devuelve (g, 256) promedios."""
        g, n = batches.shape[0], batches.shape[1]
        # tf.unstack y no batches[i]: en tensorflow-metal 1.2 el gradiente de StridedSlice se
        # reparte a todas las filas, mezclando los gradientes de todas las clases.
        augmented = tf.concat([dsa(b, s) for b, s in zip(tf.unstack(batches), seeds)], axis=0)
        feats = self.embedder(augmented, training=False)
        return tf.reduce_mean(tf.reshape(feats, [g, n, -1]), axis=1)

    @tf.function
    def step(self, real_uint8, classes, seed):
        """Un paso de DM sobre el grupo `classes`. real_uint8: (g, B, H, W, 3)."""
        self._reinit_embedder(seed)
        g = real_uint8.shape[0]
        # La misma semilla por clase para real y sintético: augmentación siamesa.
        seeds = [tf.random.experimental.stateless_fold_in(seed, 1000 + i) for i in range(g)]

        real = tf.cast(real_uint8, tf.float32) / 255.0
        real_mean = tf.stop_gradient(self._mean_features(real, seeds))

        syn = tf.gather(self.syn, classes)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(syn)
            syn_mean = self._mean_features(syn, seeds)
            loss = tf.reduce_sum(tf.square(real_mean - syn_mean))
        grad = tape.gradient(loss, syn)

        # Adam solo sobre las filas de este grupo: las demás clases no se mueven. El paper usa
        # SGD (lr 1, momentum 0.5) con una ConvNet con instance norm; con la red sin normalizar
        # de la Tabla 3 los gradientes por píxel son ~1e-5 y ese SGD casi no mueve las imágenes.
        # Adam normaliza la escala por píxel, así lr_img se lee como "cambio de píxel por paso".
        idx = classes[:, None]
        m = self.beta1 * tf.gather(self.m, classes) + (1 - self.beta1) * grad
        v = self.beta2 * tf.gather(self.v, classes) + (1 - self.beta2) * tf.square(grad)
        t = tf.gather(self.t, classes) + 1
        bc = tf.reshape(tf.sqrt(1 - self.beta2 ** t) / (1 - self.beta1 ** t), [-1, 1, 1, 1, 1])
        update = self.lr_img * bc * m / (tf.sqrt(v) + 1e-8)
        self.m.scatter_nd_update(idx, m)
        self.v.scatter_nd_update(idx, v)
        self.t.scatter_nd_update(idx, t)
        self.syn.scatter_nd_update(idx, tf.clip_by_value(syn - update, 0.0, 1.0))
        return loss

    def images(self):
        return self.syn.numpy()


def distill(cache, ipc, iterations, real_batch=64, classes_per_step=8, lr_img=0.01,
            seed=0, log_every=25, log_path=None, init="real"):
    """Destila `cache` (ClassCache) a ipc imágenes por clase. Devuelve (imágenes, etiquetas, historial, segundos)."""
    rng = np.random.default_rng(seed)
    num_classes = cache.num_classes
    if init == "real":  # converge mucho más rápido que partir de ruido
        init = np.stack([cache.sample(c, ipc, rng) for c in range(num_classes)]).astype(np.float32) / 255.0
    else:
        h, w, ch = cache.images.shape[1:]
        init = rng.uniform(0.4, 0.6, size=(num_classes, ipc, h, w, ch)).astype(np.float32)
    dm = DistributionMatcher(init, lr_img)

    history = []
    start = time.perf_counter()
    for it in range(1, iterations + 1):
        perm = rng.permutation(num_classes).astype(np.int32)
        total = 0.0
        for k in range(0, num_classes, classes_per_step):
            group = perm[k:k + classes_per_step]
            real = np.stack([cache.sample(c, real_batch, rng) for c in group])
            step_seed = rng.integers(0, 2**31 - 1, size=2).astype(np.int32)
            total += float(dm.step(real, group, step_seed))
        if it == 1 or it % log_every == 0 or it == iterations:
            elapsed = time.perf_counter() - start
            history.append({"iteration": it, "loss": total, "seconds": round(elapsed, 2)})
            print(f"[DM] iter {it:5d}/{iterations}  loss {total:.4f}  ({elapsed:.0f}s)", flush=True)

    elapsed = time.perf_counter() - start
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["iteration", "loss", "seconds"])
            writer.writeheader()
            writer.writerows(history)

    images = dm.images().reshape(-1, *init.shape[2:])
    labels = np.repeat(np.arange(num_classes, dtype=np.int32), ipc)
    return images, labels, history, elapsed


def save_synthetic(path, images, labels, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, images=images.astype(np.float16), labels=labels, meta=json.dumps(meta))


def save_grid(images, labels, path, class_names=None, cols=13):
    """Primera imagen sintética de cada clase en una cuadrícula."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, first = np.unique(labels, return_index=True)
    rows = math.ceil(len(first) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.45))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, i in zip(np.ravel(axes), first):
        ax.imshow(np.clip(images[i].astype(np.float32), 0, 1))
        title = class_names[labels[i]] if class_names else str(labels[i])
        ax.set_title(title[:14], fontsize=6)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    from src.classes import CLASSES
    from src.config import load_config, set_seed
    from src.data import ClassCache, cache_path

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--ipc", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["distill"]
    ipc = args.ipc or dcfg["ipc"]
    iterations = args.iterations or dcfg["iterations"]
    img_size = cfg["data"]["img_size"]
    set_seed(args.seed)

    cache_file = cache_path(cfg["paths"]["processed"], img_size)
    if not cache_file.exists():
        raise SystemExit(f"Falta {cache_file}. Corre primero: python -m src.data --config {args.config} --build-cache")
    cache = ClassCache(cache_file)

    tag = f"ipc{ipc}_res{img_size}"
    images, labels, _, seconds = distill(
        cache, ipc, iterations, dcfg["real_batch"], dcfg["classes_per_step"], dcfg["lr_img"],
        args.seed, dcfg["log_every"],
        log_path=Path(cfg["paths"]["runs"]) / f"distill_{tag}" / "loss.csv")

    meta = {"ipc": ipc, "img_size": img_size, "iterations": iterations, "seed": args.seed,
            "distill_seconds": round(seconds, 1)}
    out = synthetic_path(cfg["paths"]["processed"], ipc, img_size)
    save_synthetic(out, images, labels, meta)
    save_grid(images, labels, Path(cfg["paths"]["figures"]) / f"synthetic_{tag}.png", CLASSES)
    print(f"Conjunto destilado: {out} ({len(labels)} imágenes, {seconds:.0f}s)")


if __name__ == "__main__":
    main()
