"""Differentiable Siamese Augmentation (DSA; Zhao & Bilen, ICML 2021).

Todas las transformaciones son diferenciables respecto a la imagen, así el gradiente de la
pérdida de DM llega hasta los píxeles sintéticos. "Siamesa" significa que el lote real y el
sintético de una clase reciben exactamente la misma transformación: todos los parámetros se
derivan de una sola semilla y se comparten en todo el lote.
"""

import math

import tensorflow as tf

BRIGHTNESS = 0.2           # suma en [-0.2, 0.2]
SATURATION = (0.5, 1.5)    # factor multiplicativo
CONTRAST = (0.5, 1.5)
SCALE = (0.8, 1.2)
ROTATION_DEG = 15.0
TRANSLATION = 0.125        # fracción del lado de la imagen (el "crop" de DSA)


def sample_params(seed):
    """Parámetros escalares de todas las transformaciones a partir de una semilla [2]."""
    u = tf.random.stateless_uniform([8], seed=seed)
    lerp = lambda lo_hi, t: lo_hi[0] + (lo_hi[1] - lo_hi[0]) * t
    return {
        "brightness": (u[0] - 0.5) * 2.0 * BRIGHTNESS,
        "saturation": lerp(SATURATION, u[1]),
        "contrast": lerp(CONTRAST, u[2]),
        "scale": lerp(SCALE, u[3]),
        "angle": (u[4] - 0.5) * 2.0 * ROTATION_DEG * math.pi / 180.0,
        "tx": (u[5] - 0.5) * 2.0 * TRANSLATION,
        "ty": (u[6] - 0.5) * 2.0 * TRANSLATION,
        "flip": u[7] < 0.5,
    }


def _color(x, p):
    x = x + p["brightness"]
    gray = tf.reduce_mean(x, axis=-1, keepdims=True)
    x = (x - gray) * p["saturation"] + gray
    mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    return (x - mean) * p["contrast"] + mean


def _affine(x, p):
    """Escala + rotación + traslación alrededor del centro, con relleno de ceros."""
    size = tf.cast(tf.shape(x)[1:3], tf.float32)
    h, w = size[0], size[1]
    cx, cy = (w - 1.0) / 2.0, (h - 1.0) / 2.0
    cos, sin = tf.cos(p["angle"]) / p["scale"], tf.sin(p["angle"]) / p["scale"]
    # Transformación inversa: para cada píxel de salida (x, y) da la coordenada de entrada.
    a0, a1 = cos, -sin
    b0, b1 = sin, cos
    a2 = cx - a0 * cx - a1 * cy + p["tx"] * w
    b2 = cy - b0 * cx - b1 * cy + p["ty"] * h
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None]
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=x,
        transforms=tf.tile(transform, [tf.shape(x)[0], 1]),
        output_shape=tf.shape(x)[1:3],
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
    )


def dsa(x, seed):
    """Aplica color -> afín -> volteo a un lote (N, H, W, 3) con parámetros compartidos."""
    p = sample_params(seed)
    x = _color(x, p)
    x = _affine(x, p)
    return tf.cond(p["flip"], lambda: tf.reverse(x, axis=[2]), lambda: x)
