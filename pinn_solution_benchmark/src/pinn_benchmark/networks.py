"""Neural networks for the PINN benchmark. All float64.

Input normalization: affine map of (z, t) (and conditioning parameters eta)
to approximately [-1, 1], with the map stored in the module so checkpoints
are self-contained.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


def _make_activation(name: str):
    if name == "tanh":
        return nn.Tanh()
    if name == "sine":
        return Sine(w0=1.0)
    raise ValueError(f"activation {name!r} not supported (need smooth C^4)")


class FourierFeatures(nn.Module):
    """Random Fourier features (Tancik et al.), smooth, fixed at init."""

    def __init__(self, in_dim: int, n_features: int, sigma: float = 2.0):
        super().__init__()
        B = torch.randn(in_dim, n_features) * sigma
        self.register_buffer("B", B)

    @property
    def out_dim(self):
        return 2 * self.B.shape[1]

    def forward(self, x):
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class InputScaler(nn.Module):
    """x -> 2*(x - lo)/(hi - lo) - 1 per dimension."""

    def __init__(self, lo, hi):
        super().__init__()
        lo = torch.as_tensor(lo, dtype=torch.float64)
        hi = torch.as_tensor(hi, dtype=torch.float64)
        span = torch.where(hi - lo == 0, torch.ones_like(hi), hi - lo)
        self.register_buffer("lo", lo)
        self.register_buffer("span", span)

    def forward(self, x):
        return 2 * (x - self.lo) / self.span - 1


class PINN(nn.Module):
    """(z, t) [+ eta] -> (u, v). Smooth activations; supports 4th derivatives."""

    def __init__(self, hidden_layers: int = 6, width: int = 64, activation: str = "tanh",
                 fourier_features: int = 0, fourier_sigma: float = 2.0,
                 input_lo=(0.0, -1.0), input_hi=(1.0, 1.0), cond_dim: int = 0):
        super().__init__()
        self.cond_dim = cond_dim
        in_dim = 2 + cond_dim
        self.scaler = InputScaler(input_lo, input_hi)
        self.ff = None
        first_in = in_dim
        if fourier_features > 0:
            self.ff = FourierFeatures(in_dim, fourier_features, fourier_sigma)
            first_in = self.ff.out_dim
        layers = [nn.Linear(first_in, width), _make_activation(activation)]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(width, width), _make_activation(activation)]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)
        # Xavier init (tanh-appropriate)
        for mod in self.net:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight)
                nn.init.zeros_(mod.bias)

    def forward(self, z, t, eta=None):
        feats = [z, t]
        if self.cond_dim:
            if eta is None:
                raise ValueError("conditioned PINN requires eta")
            if eta.dim() == 1:
                eta = eta.unsqueeze(0).expand(z.shape[0], -1)
            feats.append(eta)
        x = torch.cat(feats, dim=-1)
        x = self.scaler(x)
        if self.ff is not None:
            x = self.ff(x)
        out = self.net(x)
        return out[:, 0:1], out[:, 1:2]

    def complex_field(self, z, t, eta=None):
        u, v = self.forward(z, t, eta)
        return torch.complex(u, v)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
