"""Adam con operadores compatibles con DirectML, sin el fallback de lerp a CPU."""

import math

import torch


class DirectMLAdam(torch.optim.Optimizer):
    """Adam denso estándar: mismos momentos y corrección de sesgo que torch.optim.Adam.

    Reemplaza m.lerp_(g, 1-beta1) por m.mul_(beta1).add_(g, alpha=1-beta1).
    Los momentos permanecen en el dispositivo del parámetro. El contador es de
    Python, así que no requiere leer un escalar desde la GPU en cada paso.
    """

    _decoupled_decay = False

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
                if self._decoupled_decay and group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
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


class DirectMLAdamW(DirectMLAdam):
    """AdamW: decay separado de los momentos, con operadores compatibles con DML."""

    _decoupled_decay = True

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-7, weight_decay=0.01):
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("AdamW requiere weight_decay finito >= 0")
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        self.defaults["weight_decay"] = weight_decay
        for group in self.param_groups:
            group.setdefault("weight_decay", weight_decay)
            if not math.isfinite(group["weight_decay"]) or group["weight_decay"] < 0:
                raise ValueError("AdamW requiere weight_decay finito >= 0")


def build_optimizer(model, train, *, native=False):
    from src.training_policy import optimizer_options

    name, decay = optimizer_options(train)
    if name == "adamw":
        # Regularizar matrices de pesos; excluir los sesgos de las capas.
        groups = [{"params": [p for p in model.parameters() if p.ndim > 1], "weight_decay": decay},
                  {"params": [p for p in model.parameters() if p.ndim <= 1], "weight_decay": 0.0}]
        cls = torch.optim.AdamW if native else DirectMLAdamW
        options = {"foreach": False} if native else {}
        return cls(groups, lr=train["lr"], eps=1e-7, weight_decay=decay, **options)
    if native:
        return torch.optim.Adam(model.parameters(), lr=train["lr"], eps=1e-7, foreach=False)
    return DirectMLAdam(model.parameters(), lr=train["lr"], eps=1e-7)
