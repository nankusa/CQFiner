import torch
from typing import Callable, Optional
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Dense(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[Callable] = None,
        bias: bool = True,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(F.linear(x, self.weight, self.bias))


def scatter_add(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = src.new_zeros((dim_size,) + src.shape[1:])
    return out.index_add(0, index, src)


class GaussianRBF(nn.Module):
    def __init__(self, n_rbf: int, cutoff: float):
        super().__init__()
        offsets = torch.linspace(0.0, cutoff, n_rbf)
        widths = (offsets[1] - offsets[0]) * torch.ones_like(offsets)
        self.register_buffer("offsets", offsets)
        self.register_buffer("widths", widths)

    def forward(self, d: Tensor) -> Tensor:
        coeff = -0.5 / (self.widths**2)
        diff = d - self.offsets
        return torch.exp(coeff * (diff**2))
