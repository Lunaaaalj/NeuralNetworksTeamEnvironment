"""Execute the deliverable notebook with this project's Python interpreter."""

import argparse
import json
from pathlib import Path
import sys

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import nbformat
from nbclient import NotebookClient

from src.config import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    kernels = output / "jupyter-kernels"
    kernel_dir = kernels / "avance4"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python del proyecto Avance 4", "language": "python",
    }), encoding="utf-8")
    path = ROOT / "notebooks/equipo_proyecto_avance4.ipynb"
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="avance4", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    client = NotebookClient(notebook, km=manager, timeout=240,
                            resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
        nbformat.write(notebook, path)
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells: {path}")


if __name__ == "__main__":
    main()
