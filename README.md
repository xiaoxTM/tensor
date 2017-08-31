# C++ Tensor Library
Tensor library is inspired by numpy and is implemented with similar API.

## Prerequisite
- [libu](https://github.com/xiaoxTM/libu) (for logging and test) 
- g++ 
- std > c++11

## Example
see @main.cpp for example usages.
```cpp
float data[] =  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
                 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                 41, 42, 43, 44, 45, 46, 47 };
float data2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                 41, 42, 43, 44, 45, 46, 47, 48};

u::tensor::Tensor t8(reinterpret_cast<unsigned char *>(data), {2,3,8}, u::tensor::DType::float32);
u::tensor::Tensor t9(reinterpret_cast<unsigned char *>(data2), {2,3,8}, u::tensor::DType::float32);
std::cout << t8 + t9 << std::endl;
std::cout << t8 + 10 << std::endl;
std::cout << 10 + t8 << std::endl;
std::cout << t8 - t9 << std::endl;
std::cout << t9 - 10 << std::endl;
std::cout << 10 - t9 << std::endl;
std::cout << t8 * t9 << std::endl;
std::cout << t8 * 10 << std::endl;
std::cout << 10 * t8 << std::endl;
std::cout << t8 / t9 << std::endl;
std::cout << t9 / 10 << std::endl;
std::cout << 10 / t9 << std::endl;
std::cout << ++ t8 << std::endl;
std::cout << t9 ++ << std::endl;
```

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
