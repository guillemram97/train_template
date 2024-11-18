import math
from typing import Optional

import torch
from torch import Tensor
from torch import nn
import numpy as np
import torch.nn.functional as F
import pdb

EPS = 1e-12


class ModularModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._task_ids = None


class LoRALinear(ModularModule):
    """Applies a linear function parameterised by a base bias
    and a weighted average of base and skill weights
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        weight: Tensor,
        bias: Optional[Tensor],
        r: int,
        lora_scaling: float,
        seed: int,
    ) -> None:
        super().__init__()

        if bias is None:
            bias = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

        #assert weight.size(0) == weight.size(1)

        D_MODEL_A = weight.size(1)
        D_MODEL_B = weight.size(0)

        self.r = r

        self.weight = nn.Parameter(weight.data, requires_grad=False)

        total_shape_A = (r, D_MODEL_A,)
        total_shape_B = (D_MODEL_B, r)
        self.device = self.weight.device

        self.bias = nn.Parameter(bias.data, requires_grad=False).to(self.device)
        self.A = nn.Parameter(torch.zeros(total_shape_A), requires_grad=True).to(self.device)
        self.B = nn.Parameter(torch.zeros(total_shape_B), requires_grad=True).to(self.device)

        self.scaling = lora_scaling
        self.reset_parameters()

    def reset_parameters(self):
        # We init in such a way to have A*B equal to 0 at first
        # It's crucial for convergence
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, input: Tensor) -> Tensor:
        """
        input: [batch_size, seq_length, input_features]
        """
        # I SHOULD REVISE ALL OF THIS
        AB = torch.einsum("or,ri->oi", self.B, self.A)
        output = torch.einsum("oi, bni->bno", AB, input)
        F.linear(input, self.weight, self.bias)
        output = F.linear(input, self.weight, self.bias) + output * self.scaling
        return output
