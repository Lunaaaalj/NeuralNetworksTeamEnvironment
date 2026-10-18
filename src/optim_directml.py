"""Adam con operadores compatibles con DirectML, sin el fallback de lerp a CPU."""

import math

import torch


class DirectMLAdam(torch.optim.Optimizer):
    """Adam denso estándar: mismos momentos y corrección de sesgo que torch.optim.Adam.

    Reemplaza m.lerp_(g, 1-beta1) por m.mul_(beta1).add_(g, alpha=1-beta1).
    Los momentos permanecen en el dispositivo del parámetro. El contador es de
    Python, así que no requiere leer un escalar desde la GPU en cada paso.
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-7):
        if lr < 0 or eps < 0 or not all(0 <= beta < 1 for beta in betas):
            raise ValueError("Adam requiere lr/eps >= 0 y betas en [0, 1).")
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse or torch.is_complex(parameter):
                    raise RuntimeError("DirectMLAdam admite gradientes densos y parámetros reales.")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                first, second = state["exp_avg"], state["exp_avg_sq"]
                first.mul_(beta1).add_(gradient, alpha=1 - beta1)
                second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                denominator = (second.sqrt() / math.sqrt(1 - beta2 ** state["step"])).add_(group["eps"])
                parameter.addcdiv_(first, denominator, value=-group["lr"] / (1 - beta1 ** state["step"]))
        return loss
