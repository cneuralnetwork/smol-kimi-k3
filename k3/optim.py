from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def zeroth_power_via_newton_schulz(
    gradient: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Polar-factor approximation used by Muon; supports batched head matrices."""
    x = gradient.float()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.transpose(-2, -1)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    # Quintic iteration coefficients used by the public Muon implementation.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = x @ x.transpose(-2, -1)
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.transpose(-2, -1)
    return x.to(gradient.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.01,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(gradient)
                buffer = state["momentum_buffer"]
                buffer.lerp_(gradient, 1.0 - group["momentum"])
                update = gradient.lerp(buffer, group["momentum"])

                heads = getattr(parameter, "_muon_heads", 1)
                if heads > 1 and update.shape[0] % heads == 0:
                    head_shape = (heads, update.shape[0] // heads, update.shape[1])
                    update = zeroth_power_via_newton_schulz(
                        update.view(head_shape), group["ns_steps"]
                    ).view_as(update)
                    rows, columns = head_shape[-2:]
                else:
                    update = zeroth_power_via_newton_schulz(update, group["ns_steps"])
                    rows, columns = update.shape[-2:]
                update *= math.sqrt(max(1.0, rows / columns))

                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        return loss


class K3Optimizers:
    """Muon for matrices and AdamW for embeddings, norms, biases, and scalars."""

    def __init__(
        self,
        model: torch.nn.Module,
        muon_lr: float,
        adamw_lr: float,
        weight_decay: float,
    ) -> None:
        muon_params: list[torch.nn.Parameter] = []
        adamw_decay: list[torch.nn.Parameter] = []
        adamw_no_decay: list[torch.nn.Parameter] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim == 2 and "token_embedding" not in name and "lm_head" not in name:
                muon_params.append(parameter)
            elif parameter.ndim >= 2:
                adamw_decay.append(parameter)
            else:
                adamw_no_decay.append(parameter)

        self.muon = Muon(muon_params, lr=muon_lr, weight_decay=weight_decay)
        self.adamw = torch.optim.AdamW(
            [
                {"params": adamw_decay, "weight_decay": weight_decay},
                {"params": adamw_no_decay, "weight_decay": 0.0},
            ],
            lr=adamw_lr,
            betas=(0.9, 0.95),
            fused=torch.cuda.is_available(),
        )
        self.base_lrs = (muon_lr, adamw_lr)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.muon.step()
        self.adamw.step()

    def set_lr_scale(self, scale: float) -> None:
        for group in self.muon.param_groups:
            group["lr"] = self.base_lrs[0] * scale
        for group in self.adamw.param_groups:
            group["lr"] = self.base_lrs[1] * scale

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])
