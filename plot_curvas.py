"""Grafica las curvas de entrenamiento vs validación a partir de history.csv
e imprime los números que van en la sección de Regularización.

Ponlo en scripts/plot_curvas.py del repo y córrelo desde la raíz:
    python scripts/plot_curvas.py                                   # corrida full_res112_s0
    python scripts/plot_curvas.py reports/runs/random_ipc1_res112_s0/history.csv
Genera la figura en reports/figures/ (main.tex ya busca ahí con \graphicspath).
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ruta = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/runs/full_res112_s0/history.csv")
df = pd.read_csv(ruta)
print("Columnas encontradas:", list(df.columns))


def buscar(*claves, excluir=()):
    """Regresa la primera columna que contiene todas las claves (sin las excluidas)."""
    for c in df.columns:
        n = c.lower()
        if all(k in n for k in claves) and not any(e in n for e in excluir):
            return c
    return None


ep = buscar("epoch")
x = df[ep] if ep else pd.Series(range(len(df)))
if x.min() == 0:          # CSVLogger numera desde 0; en el reporte las épocas van desde 1
    x = x + 1
tl = buscar("train", "loss") or buscar("loss", excluir=("val",))
vl = buscar("val", "loss")
ta = buscar("train", "acc") or buscar("acc", excluir=("val",))
va = buscar("val", "acc")
vf = buscar("f1")
print(f"Usando -> época: {ep}, train_loss: {tl}, val_loss: {vl}, "
      f"train_acc: {ta}, val_acc: {va}, val_f1: {vf}")
print("Si alguna salió None o equivocada, cambia el nombre a mano en estas líneas.\n")

# Si la exactitud viene en 0-1, pasarla a porcentaje
for c in (ta, va):
    if c and df[c].max() <= 1.0:
        df[c] = df[c] * 100

i_best = df[vf].idxmax() if vf else df[va].idxmax()
i_vlmin = df[vl].idxmin()
ep_best = 





fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].plot(x, df[tl], label="Entrenamiento")
val = df[vl].notna()      # en conjuntos pequeños val solo se mide cada 10 épocas
ax[0].plot(x[val], df[vl][val], "o-", ms=3, label="Validación")
ax[0].set_title("Pérdida")
ax[1].plot(x, df[ta], label="Entrenamiento")
ax[1].plot(x[val], df[va][val], "o-", ms=3, label="Validación")
ax[1].set_title("Exactitud (%)")
for a in ax:
    a.axvline(ep_best, ls="--", c="gray", lw=1)
    a.set_xlabel("Época")
    a.grid(alpha=0.3)
    a.legend()
plt.tight_layout()
nombre = "curvas_full.png" if "full" in ruta.parent.name else f"curvas_{ruta.parent.name}.png"
salida = Path("reports/figures") / nombre
salida.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(salida, dpi=200)
print(f"Figura guardada: {salida}\n")

print("=== Números para el .tex ===")
print(f"Época con mejor F1 macro:          {ep_best}")
print(f"Época con val_loss mínima:         {x[i_vlmin]}  (val_loss = {df[vl][i_vlmin]:.4f})")
print(f"train_loss en la última época:     {df[tl].iloc[-1]:.4f}")
print(f"train_acc en la mejor época:       {df[ta][i_best]:.2f} %")
print(f"train_acc en la última época:     {df[ta].iloc[-1]:.2f} %   (para IPC = 1)")
print(f"val_acc en la mejor época:         {df[va][i_best]:.2f} %")
print(f"Brecha (train - val) mejor época:  {df[ta][i_best] - df[va][i_best]:.2f} puntos")
print(f"Brecha en la última época:         {df[ta].iloc[-1] - df[va].iloc[-1]:.2f} puntos")
