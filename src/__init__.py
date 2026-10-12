import sys

# TF 2.18 (la versión compatible con tensorflow-metal) hace `import distutils`, que en
# Python 3.12 provee setuptools mediante un .pth. Python >= 3.12.10 ignora los .pth que
# macOS marca como ocultos, así que se activa el shim importando setuptools explícitamente.
try:
    import setuptools  # noqa: F401
except ImportError:
    pass

if sys.platform == "darwin":
    # tensorflow-metal 1.2 fusiona MatMul + BiasAdd + Relu en modo grafo y el kernel fusionado
    # omite la ReLU: fit/predict entrenan y evalúan una capa densa lineal mientras que la
    # llamada eager sí aplica ReLU. Apagar el meta-optimizador de Grappler evita la fusión
    # sin costo de velocidad apreciable. Ver tests/test_model.py::test_graph_matches_eager.
    import tensorflow as tf

    tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
