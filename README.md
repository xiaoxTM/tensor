# C++ Tensor Library
Tensor library is inspired by numpy and is implemented with similar API.

## Example
see @main.cpp for example usages.

## Workflow


```
u-tensor.hpp --> u-op.hpp --> u-cpu-op.hpp
                     |
                     V
                u-gpu-op.hpp
```

**Call Process Example** [tensor.add]
```
add (function) --> u::op::Add (class) --> u::op::cpu::add (function) --> u::aop::Add (class)
                        |
                        V
                u::op::gpu::add (function)
```

## ToDo

- [ ] Expression template to implement effectively element-wise operation. Such as a = b + c + d + e + f
- [ ] BLAS introduction
- [ ] GPU `CUDA` implementation
- [ ] Indexing
