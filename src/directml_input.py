"""Entrega los mismos lotes de tf.data a PyTorch sin copiarlos a NumPy."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
import torch


INPUT_PIPELINES = ("numpy", "dlpack", "dlpack_prefetch")


def host_batches(dataset, pipeline):
    """Solo lectura de los buffers compartidos; NCHW se prepara en la CPU."""
    for images, labels in dataset:
        if pipeline == "numpy":
            images = torch.from_numpy(images.numpy())
            labels = torch.from_numpy(labels.numpy().astype(np.int64, copy=False))
        else:
            images = torch.from_dlpack(tf.experimental.dlpack.to_dlpack(images))
            labels = torch.from_dlpack(tf.experimental.dlpack.to_dlpack(labels)).to(torch.int64)
        yield images.permute(0, 3, 1, 2).contiguous(), labels


def batches(dataset, device, *, pipeline="dlpack_prefetch"):
    """Anticipa como máximo un lote de CPU; las llamadas DirectML van en el hilo principal.

    numpy conserva el puente anterior para comparaciones. DLPack comparte la
    memoria de TensorFlow con PyTorch; el cambio a NCHW y la copia a GPU siguen
    siendo necesarios. Los aumentos y el mezclado permanecen en tf.data.
    """
    if pipeline not in INPUT_PIPELINES:
        raise ValueError(f"runtime.input_pipeline debe ser uno de {INPUT_PIPELINES}")
    source = host_batches(dataset, pipeline)
    if pipeline != "dlpack_prefetch":
        for images, labels in source:
            yield images.to(device), labels.to(device)
        return
    # El contexto espera al trabajador también ante excepciones o al cerrar el
    # generador: no deja lectores de TFRecords huérfanos entre corridas.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="directml-input") as pool:
        pending = pool.submit(next, source, None)
        try:
            while True:
                batch = pending.result()
                if batch is None:
                    break
                pending = pool.submit(next, source, None)
                images, labels = batch
                yield images.to(device), labels.to(device)
        finally:
            pending.cancel()
