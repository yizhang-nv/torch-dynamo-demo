import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, subgraph_rewriter, symbolic_trace

from backend import BackendClass


class Config:

    def __init__(self, mlp, size):
        self.mlp = mlp
        self.size = size


# Pretend that this is an optimized custom op
@torch.library.custom_op("custom_op::high_performance_op", mutates_args=())
def gemm_ln_gemm(
    x: torch.Tensor, l0_weight: torch.Tensor, l1_wight: torch.Tensor
) -> torch.Tensor:
    return F.linear(F.linear(x, l0_weight), l1_wight)


class CustomLinearModule(torch.nn.Module):
    def __init__(self, in_size, out_size=None):
        super().__init__()
        if out_size == None:
            out_size = in_size
        self.weight = torch.nn.Parameter(torch.rand((out_size, in_size)))

    def forward(self, x):
        return F.linear(x, self.weight)


def source_pattern(x, config: Config, proj, down_proj):
    return CustomModel.forward_pattern(x, config, proj, down_proj)


def target_pattern(x, config: Config, proj, down_proj):
    return x


def fn(gm: GraphModule, example_inputs):

    source_pattern_fx = symbolic_trace(
        source_pattern, concrete_args={"config": Config(True, 1)}
    )

    # Match fail
    # AssertionError: SubgraphMatcher cannot be initialized with an pattern with dead code
    subgraph_rewriter.replace_pattern(gm, source_pattern_fx, target_pattern)

    print("After optimization: ")
    gm.graph.print_tabular()
    gm.recompile()

    return gm.forward


custom_backend = BackendClass(fn)


class CustomModel(torch.nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        if config.mlp:
            self.up_proj = CustomLinearModule(config.size, config.size * 3)
            self.down_proj = CustomLinearModule(config.size * 3, config.size)
        else:
            self.proj = CustomLinearModule(config.size)
        self.config = config

    def forward(self, x):
        return CustomModel.forward_pattern(
            x,
            self.config,
            self.up_proj if config.mlp else self.proj,
            self.down_proj if config.mlp else None,
        )

    @staticmethod
    def forward_pattern(x, config: Config, proj, down_proj):
        residual = x
        if config.mlp:
            x = down_proj(proj(x))
        else:
            x = proj(x)

        return x + residual


config = Config(True, 100)
model = CustomModel(config)
model_opt = torch.compile(model, backend=custom_backend, fullgraph=True)

x = torch.rand((8, 100))
ref = model(x)
opt = model_opt(x)

torch.testing.assert_close(ref, opt)
