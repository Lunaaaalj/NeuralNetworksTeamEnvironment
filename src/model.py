"""CNN propuesta en la Tabla 3 del reporte y el extractor de características usado por DM."""

import keras
from keras import layers

FILTERS = (32, 64, 128, 256)
EXTRA_CONV_STAGES = {"baseline": (), "deep": (3, 4), "batchnorm": ()}
ARCHITECTURES = (*EXTRA_CONV_STAGES, "mlp", "efficientnetb0")
BATCHNORM_EPSILON = 1e-3
BATCHNORM_MOMENTUM = 0.9  # Keras: proporción conservada de la media/varianza anterior.


def model_architecture(cfg):
    """La CNN original sigue siendo la opción por defecto de configuraciones antiguas."""
    architecture = cfg.get("model", {}).get("architecture", "baseline")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"model.architecture debe ser uno de {ARCHITECTURES}")
    return architecture


def _regularizer(l2):
    return keras.regularizers.L2(l2) if l2 else None


def _conv_stack(x, architecture="baseline", l2=0.0):
    """Cuatro poolings, con convoluciones adicionales o BatchNorm según la variante."""
    if architecture not in EXTRA_CONV_STAGES:
        raise ValueError(f"Arquitectura desconocida: {architecture}")
    for i, f in enumerate(FILTERS, start=1):
        normalized = architecture == "batchnorm"
        x = layers.Conv2D(f, 3, padding="same", activation=None if normalized else "relu",
                          use_bias=not normalized, kernel_regularizer=_regularizer(l2), name=f"conv{i}")(x)
        if normalized:
            x = layers.BatchNormalization(axis=-1, epsilon=BATCHNORM_EPSILON,
                                           momentum=BATCHNORM_MOMENTUM, name=f"bn{i}")(x)
            x = layers.ReLU(name=f"relu{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
        if i in EXTRA_CONV_STAGES[architecture]:
            x = layers.Conv2D(f, 3, padding="same", activation="relu",
                              kernel_regularizer=_regularizer(l2), name=f"conv{i}_extra")(x)
    return layers.GlobalAveragePooling2D(name="gap")(x)


def build_cnn(num_classes=104, input_shape=(None, None, 3), dropout=0.5, *, architecture="baseline", l2=0.0):
    """CNN de la Tabla 3 o variantes deep/batchnorm. Devuelve logits.

    La entrada es de tamaño espacial libre: gracias a GAP la misma red sirve a 224x224
    o a 112x112 (resolución usada para destilar en local). l2 > 0 penaliza los kernels
    (no los sesgos) dentro de la pérdida.
    """
    inputs = keras.Input(shape=input_shape, name="image")
    x = _conv_stack(inputs, architecture, l2)
    x = layers.Dense(256, activation="relu", kernel_regularizer=_regularizer(l2), name="dense")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    logits = layers.Dense(num_classes, kernel_regularizer=_regularizer(l2), name="logits")(x)
    name = "flower_cnn" if architecture == "baseline" else f"flower_cnn_{architecture}"
    return keras.Model(inputs, logits, name=name)


def build_mlp(num_classes=104, img_size=112, dropout=0.5, hidden=256):
    """Línea base totalmente conectada: imagen aplanada -> Dense/ReLU -> Dropout -> logits.

    Misma cabeza que la CNN, pero sin convoluciones: cada píxel es una entrada independiente.
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Flatten(name="flatten")(inputs)
    x = layers.Dense(hidden, activation="relu", name="dense")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    logits = layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs, logits, name="flower_mlp")


def build_efficientnet(num_classes=104, img_size=224, dropout=0.2, weights="imagenet"):
    """EfficientNetB0 preentrenada en ImageNet + cabeza nueva. Devuelve logits.

    Nuestras imágenes llegan en [0, 1]; EfficientNet de Keras normaliza internamente
    entradas en [0, 255], de ahí el Rescaling. La red base se llama "backbone" para
    poder congelarla y descongelarla por fases.
    """
    backbone = keras.applications.EfficientNetB0(include_top=False, weights=weights,
                                                 input_shape=(img_size, img_size, 3), pooling="avg")
    backbone.name = "backbone"
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Rescaling(255.0, name="to_0_255")(inputs)
    x = backbone(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    logits = layers.Dense(num_classes, name="logits")(x)
    return keras.Model(inputs, logits, name="flower_efficientnetb0")


def build_model(cfg):
    """Construye la arquitectura de cfg['model'] (por defecto, la CNN original con dropout 0.5)."""
    architecture = model_architecture(cfg)
    options = cfg.get("model", {})
    num_classes, img_size = cfg["data"]["num_classes"], cfg["data"]["img_size"]
    if architecture == "mlp":
        return build_mlp(num_classes, img_size, options.get("dropout", 0.5))
    if architecture == "efficientnetb0":
        return build_efficientnet(num_classes, img_size, options.get("dropout", 0.2))
    return build_cnn(num_classes, dropout=options.get("dropout", 0.5), architecture=architecture,
                     l2=options.get("l2", 0.0))


@keras.saving.register_keras_serializable(package="flowers")
class SparseSmoothedCrossentropy(keras.losses.Loss):
    """Entropía cruzada con etiquetas enteras y label smoothing (la versión sparse de Keras no lo admite)."""

    def __init__(self, smoothing=0.1, num_classes=104, name="sparse_smoothed_crossentropy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.smoothing, self.num_classes = smoothing, num_classes

    def call(self, y_true, y_pred):
        y_true = keras.ops.one_hot(keras.ops.cast(keras.ops.reshape(y_true, [-1]), "int32"), self.num_classes)
        return keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True,
                                                     label_smoothing=self.smoothing)

    def get_config(self):
        return {**super().get_config(), "smoothing": self.smoothing, "num_classes": self.num_classes}


def build_embedder(input_shape=(None, None, 3)):
    """Solo la pila convolucional hasta GAP: vector de 256 características por imagen."""
    inputs = keras.Input(shape=input_shape, name="image")
    return keras.Model(inputs, _conv_stack(inputs), name="embedder")


def compile_cnn(model, lr=1e-3, *, optimizer="adam", weight_decay=0.0, label_smoothing=0.0):
    from src.training_policy import optimizer_options

    name, decay = optimizer_options({"optimizer": optimizer, "weight_decay": weight_decay})
    if name == "adamw":
        optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay=decay)
        optimizer.exclude_from_weight_decay(var_names=["bias"])
    else:
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    if label_smoothing:
        loss = SparseSmoothedCrossentropy(label_smoothing, model.output_shape[-1])
    else:
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model
