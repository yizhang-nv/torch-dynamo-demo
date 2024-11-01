import time
from functools import wraps
from typing import Callable, List

import tensorrt_llm
import torch
from torch._subclasses import FakeTensor
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import ShapeEnv


def time_it(func=None, *, backend=None):

    if func == None:

        def fn(func: Callable):
            if func is None:
                raise RuntimeError("func can't be None")
            return time_it(func, backend=backend)

        return fn

    @wraps(func)
    def func_wrapper(*args, **kwargs):
        if backend is None:
            start = time.perf_counter()
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        result = func(*args, **kwargs)
        if backend is None:
            end = time.perf_counter()
            self: BackendClass = args[0]
            assert isinstance(self, BackendClass)
            elapsed_time = end - start
            self.elapsed_time += elapsed_time
        else:
            end.record()
            self = backend
            assert isinstance(self, BackendClass)
            self.module_inference_event.append((start, end))
        return result

    return func_wrapper


class Timer:

    def __init__(self):
        self.s = torch.cuda.Event(enable_timing=True)
        self.e = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.s.record()

    def end(self):
        self.e.record()

    def get_elapsed_time(self):
        torch.cuda.synchronize()
        return self.s.elapsed_time(self.e) / 1000


class BackendClass:

    def __init__(self, optimize_fn=lambda gm, example_inputs: gm.forward) -> None:
        super().__init__()
        self.optimize_fn = optimize_fn
        self.elapsed_time = 0
        self.module_inference_event = []
        self.module_inference_time = 0
        self.call_count = 0

    @time_it
    def __call__(self, gm: GraphModule, example_inputs: List[torch.Tensor]) -> callable:
        print("my_compiler() called with FX graph:")
        self.call_count += 1
        rank = tensorrt_llm.mpi_rank()
        if rank == 0:
            gm.graph.print_tabular()

            def contains_sym_int(shape):
                return any(isinstance(dim, torch.SymInt) for dim in shape)

            for node in gm.graph.nodes:
                print(node, node.op)
                if node.op != "output":
                    if "example_value" in node.meta:
                        tensor = node.meta["example_value"]
                    else:
                        print(node.meta)
                        continue

                    if isinstance(tensor, FakeTensor) and contains_sym_int(
                        tensor.shape
                    ):
                        print(f"Dynamic shape detected for {node}.")

                        for idx, i in enumerate(tensor.shape):
                            print(f"Dim {idx} ", end="")
                            if isinstance(i, torch.SymInt):
                                node = i.node
                                expr = node.expr
                                shape_env: ShapeEnv = node.shape_env
                                var_range = shape_env.var_to_range.get(
                                    expr, None
                                ) or shape_env.bound_sympy(expr)
                                var_val = shape_env.var_to_val.get(
                                    expr, None
                                ) or expr.xreplace(shape_env.var_to_val)
                                print(
                                    f"(Dynamic): min {var_range.lower} opt {var_val} max {var_range.upper}"
                                )
                            else:
                                print(f"(Static): {i}")

        backend_compiled = self.optimize_fn(gm, example_inputs)

        result = gm.forward
        if backend_compiled is not None:
            result = backend_compiled

        return time_it(result, backend=self)

    def get_elapsed_time(self):
        for s, e in self.module_inference_event:
            self.module_inference_time += s.elapsed_time(e) / 1000
        return self.elapsed_time, self.module_inference_time

    def clear_time(self):
        self.elapsed_time = 0
        self.module_inference_time = 0
        self.module_inference_event.clear()
