# Replace Pattern Issues

This folder contains several minimum reproduce code to demonstrate the obstacle we've encontered when using `replace_pattern` to do the high level graph rewriting.

## Replace Module

We've found that for either pytorch built-in module and custom module, we cannot easily match and rewrite them. This is crucial since we sometimes want to fuse multiple modules into one custom op, and it would be nice if `replace_pattern` could support it. 

### Built-in Module

For built-in module, it seems that we cannot directly match it. Example code: [replace_builtin_module.py](https://github.com/yizhang-nv/torch-dynamo-demo/blob/main/replace_builtin_module.py?ref_type=heads). 

Ref: https://discuss.pytorch.org/t/torch-fx-replace-modules/116315, https://discuss.pytorch.org/t/replace-pattern-for-nn-modules/123399

And we've found that kernl hacked the `replace_pattern` themselves to achieve it: https://github.com/ELS-RD/kernl/blob/main/src/kernl/utils/extended_matcher.py#L460. Is there an official support for this in the future? 

### Custom Module

For custom module, it seems that `replace_pattern` itself cannot register the module's parameters since it cannot handle the character `.` in the parameter's name. Example code: [replace_custom_module.py](https://github.com/yizhang-nv/torch-dynamo-demo/blob/main/replace_custom_module.py?ref_type=heads)

```
torch._dynamo.exc.BackendCompilerFailed: backend='<backend.BackendClass object at 0x7f1429ea4820>' raised:
KeyError: 'parameter name can\'t contain "."'
```

The root cause maybe torch currently does not have a mapping system to map the parameters in the target pattern to the source pattern so that the rewrote graph cannot get the correct parameter in the original graph. So emitting such error is a reasonable way. 

### WAR

There is an workaround for the above issue: change module's forward to a stateless function. Example code: [replace_module_war.py](https://github.com/yizhang-nv/torch-dynamo-demo/blob/main/replace_module_war.py?ref_type=heads)

```python
class CustomLinearModule(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((size, size)))

    def forward(self, x):
        return CustomLinearModule.forward_pattern(x, self.weight)

    @staticmethod
    def forward_pattern(x, weight):
        return F.linear(x, weight)
```

And define the pattern with that static forward function (`forward_pattern` in the above example).

```python

class CustomModel(torch.nn.Module):

    def __init__(self, size):
        super().__init__()
        self.l0 = CustomLinearModule(size)
        self.l1 = CustomLinearModule(size)

    def forward(self, x):
        residual = x
        x = self.l1(self.l0(x)) # We want to match this.
        return x + residual

def source_pattern(x, l0_weight, l1_weight):
    return CustomLinearModule.forward_pattern( # Use staticmethod version of forward here.
        CustomLinearModule.forward_pattern(x, l0_weight), l1_weight
    )
```

However, this would require developers to rewrite all their modules to make them compatible with `replace_pattern`. Such constraints would be a burden for us to match custom modules that are not written in this way. 

Is there a better way to match both the built-in module and the custom module with `replace_pattern`?

## Dead code in Pattern Input

This is a known issue for pytorch: https://github.com/pytorch/pytorch/issues/100419. However, this could be a problem if we pass in custom objects as arguments to configure the network. Example code: [dead_code.py](https://github.com/yizhang-nv/torch-dynamo-demo/blob/main/dead_code.py?ref_type=heads)

```python
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

def source_pattern(x, config: Config, proj, down_proj):
    return CustomModel.forward_pattern(x, config, proj, down_proj)
```

Say we have a `source_pattern` that has a control flow that uses input as the condition. We can partially specialize the pattern with `source_pattern_fx = symbolic_trace(source_pattern, concrete_args={"config":Config(True, 1)})`. However, `replace_pattern` would complain that there is dead code in the source pattern since the graph is specialized and no one would use `config` anymore. 

Of course, we can move config into `source_pattern`, and specialize it there rather than using concrete args. 

```python
def source_pattern(x, proj, down_proj):
    config = Config(True, 100)
    return CustomModel.forward_pattern(x, config, proj, down_proj)
```

However, it would require us to write multiple source pattern codes for multiple configurations. 

If `replace_pattern` can allow dead code in input, then we would only maintain one source pattern code and specialize it with different input args. Also, it would make us manage the optimization pass easier.

One major concern for allowing the dead code in input would be that the target pattern may use the unused input. We can add dead code detection for the target pattern as well. If the target pattern doesn't use the dead input, then everything should be fine. 

## Hard to Specialize with Input

Although we can trace the pattern ourselves with concrete args, it is still hard for us to get the same graph as dynamo since dynamo can specialize fx graph with input attributes. Example code: [specialize_with_input.py](https://github.com/yizhang-nv/torch-dynamo-demo/blob/main/specialize_with_input.py?ref_type=heads)

```python

def func(x, y):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    x = x+y.to(torch.float32)
    return x.to(input_dtype)
```

For example, dynamo would provide us with the following op code with `x.dtype`:

```
opcode         name    target                   args                   kwargs
-------------  ------  -----------------------  ---------------------  --------
placeholder    l_x_    L_x_                     ()                     {}
placeholder    l_y_    L_y_                     ()                     {}
call_method    x       to                       (l_x_, torch.float32)  {}
call_method    to_1    to                       (l_y_, torch.float32)  {}
call_function  x_1     <built-in function add>  (x, to_1)              {}
call_method    to_2    to                       (x_1, torch.bfloat16)  {} # Specialized
output         output  output                   ((to_2,),)             {}
```

However, if we trace the graph with `torch.fx`, then it is hard to get a specialized graph.

```
opcode         name       target                       args                kwargs
-------------  ---------  ---------------------------  ------------------  --------
placeholder    x          x                            ()                  {}
placeholder    y          y                            ()                  {}
call_method    to         to                           (x, torch.float32)  {}
call_method    to_1       to                           (y, torch.float32)  {}
call_function  add        <built-in function add>      (to, to_1)          {}
call_function  getattr_1  <built-in function getattr>  (x, 'dtype')        {}
call_method    to_2       to                           (add, getattr_1)    {} # Not Specialized
output         output     output                       (to_2,)             {}
```

Given the callable with the same code, it makes us wonder how to get the dynamo version of specialized fx graph ourselves to do the correct pattern matching? Since with custom backend, we can get example inputs, so in thereory we can get the specialized graph. 

Tracing the graph with concrete args sometimes over-specialize the graph. 

## Misc & Usability

The development experience of `replace_pattern` is not ideal. 

1. Hard to get the fx graph traced by `replace_pattern`. 
2. No log for the matching process. Could only try from small graph to see which part of the pattern breaks the matching.

Also, is the fx graph traced by torch.fx/torch.dynamo version sensitive? (i.e. same python code would produce same fx graph across different pytorch versions)
