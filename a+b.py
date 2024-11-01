import torch
from torch.fx import GraphModule, replace_pattern

from backend import BackendClass


# Pretend that this is an optimized kernel
@torch.library.custom_op("custom_op::some_op", mutates_args=())
def some_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x * x
    x = x + y
    x = x * x
    return x


@some_op.register_fake
def _(x: torch.Tensor, y: torch.Tensor):
    return torch.empty_like(x)


def func(x, y):
    x = x * x
    x = x + y
    x = x * x
    return x


def source_pattern(x, y):
    x = x * x
    x = x + y
    x = x * x
    return x


def source_pattern_1(x, y):
    x = x * x
    x = y + x  # Changed here
    x = x * x
    return x


def target_pattern(x, y):
    return some_op(x, y)


def optimize(gm: GraphModule, example_input):
    replace_pattern(gm, source_pattern, target_pattern)

    gm.recompile()
    gm.graph.print_tabular()

    return gm.forward


def optimize_1(gm: GraphModule, example_input):
    replace_pattern(gm, source_pattern_1, target_pattern)

    gm.recompile()
    gm.graph.print_tabular()

    return gm.forward


x = torch.rand((1, 1), dtype=torch.bfloat16)
y = torch.rand((1, 1), dtype=torch.float16)

custom_backend_0 = BackendClass(optimize)
custom_backend_1 = BackendClass(optimize_1)

func_opt_0 = torch.compile(func, backend=custom_backend_0)
func_opt_1 = torch.compile(func, backend=custom_backend_1)

print("Run 0:")
func_opt_0(x, y)

print("Run 1:")
func_opt_1(x, y)
