import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.fx import symbolic_trace

from backend import BackendClass


def func(x, y):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + y.to(torch.float32)
    return x.to(input_dtype)


x = torch.rand((1, 1), dtype=torch.bfloat16)
y = torch.rand((1, 1), dtype=torch.float16)

custom_backend = BackendClass()

func_opt = torch.compile(func, backend=custom_backend)

func_opt(x, y)

gm = symbolic_trace(func)
gm.graph.print_tabular()

gm_concrete = symbolic_trace(func, concrete_args={"x": x})
gm.graph.print_tabular()
