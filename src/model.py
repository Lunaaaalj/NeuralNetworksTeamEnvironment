"""CNN propuesta en la Tabla 3 del reporte y el extractor de características usado por DM."""

import keras
from keras import layers

FILTERS = (32, 64, 128, 256)


def _conv_stack(x):
    """4 bloques Conv 3x3 (padding same, ReLU) + MaxPool 2x2, seguidos de Global Average Pooling."""
    for i, f in enumerate(FILTERS, start=1):
        x = layers.Conv2D(f, 3, padding="same", activation="relu", name=f"conv{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    return layers.GlobalAveragePooling2D(name="gap")(x)


def build_cnn(num_classes=104, input_shape=(None, None, 3), dropout=0.5):
    """Red completa de la Tabla 3. Devuelve logits (sin softmax).

    La entrada es de tamaño espacial libre: gracias a GAP la misma red sirve a 224x224
    o a 112x112 (resolución usada para destilar en local).
    """
    inputs = keras.Input(shape=input_shape, name="image")
    x = _conv_stack(inputs)
    x = layers.Dense(256, activation="relu", name="dense")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    logits = layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs, logits, name="flower_cnn")


def build_embedder(input_shape=(None, None, 3)):
    """Solo la pila convolucional hasta GAP: vector de 256 características por imagen."""
    inputs = keras.Input(shape=input_shape, name="image")
    return keras.Model(inputs, _conv_stack(inputs), name="embedder")


def compile_cnn(model, lr=1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model
