import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, subgraph_rewriter

from backend import BackendClass


# Pretend that this is an optimized back2back gemm kernel
@torch.library.custom_op("custom_op::gemm_gemm", mutates_args=())
def gemm_gemm(
    x: torch.Tensor, l0_weight: torch.Tensor, l1_wight: torch.Tensor
) -> torch.Tensor:
    return F.linear(F.linear(x, l0_weight), l1_wight)


class CustomLinearModule(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand((size, size)))

    def forward(self, x):
        return CustomLinearModule.forward_pattern(x, self.weight)

    @staticmethod
    def forward_pattern(x, weight):
        return F.linear(x, weight)


def source_pattern(x, l0_weight, l1_weight):
    return CustomLinearModule.forward_pattern(
        CustomLinearModule.forward_pattern(x, l0_weight), l1_weight
    )


def target_pattern(x, l0_weight, l1_weight):
    return gemm_gemm(x, l0_weight, l1_weight)


def fn(gm: GraphModule, example_inputs):

    # Match and replace successfully
    subgraph_rewriter.replace_pattern(gm, source_pattern, target_pattern)

    print("After optimization: ")
    gm.graph.print_tabular()
    gm.recompile()

    return gm.forward


custom_backend = BackendClass(fn)


class CustomModel(torch.nn.Module):

    def __init__(self, size):
        super().__init__()
        self.l0 = CustomLinearModule(size)
        self.l1 = CustomLinearModule(size)

    def forward(self, x):
        residual = x
        x = self.l1(self.l0(x))
        return x + residual


model = CustomModel(100)
model_opt = torch.compile(model, backend=custom_backend, fullgraph=True)

x = torch.rand((8, 100))
ref = model(x)
opt = model_opt(x)

torch.testing.assert_close(ref, opt)
