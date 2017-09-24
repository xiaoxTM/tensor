#ifndef __U_TENSOR_OPERATION_HPP__
#define __U_TENSOR_OPERATION_HPP__

#include <libu/u-log>

#include "u-dtype.hpp"

#ifdef USE_CUDA
#include "op/u-op-gpu.cu"
#else
#include "op/u-op-cpu.hpp"
#endif

namespace u {
    namespace tensor {
        namespace op {

            template<class C, template<typename > class Fun, class ...Args>
            void run(C &src, Args &&...args) {
                switch (src.type()) {
                    case u::tensor::DType::int8: Fun<char>::run(src.ref(), src.shape(), std::forward<Args>(args)...);break;
                    case u::tensor::DType::uint8: Fun<unsigned char>::run(src.ref(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::int16: Fun<short>::run(src.ref(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint16: Fun<unsigned short>::run(src.ref(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::int32: Fun<int>::run(src.ref(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint32: Fun<unsigned int>::run(src.ref(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::int64: Fun<long>::run(src.ref(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint64: Fun<unsigned long>::run(src.ref(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::float32: Fun<float>::run(src.ref(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::float64: Fun<double>::run(src.ref(), src.shape(), std::forward<Args>(args)...); break;
                    default:
                    bool NON_SUPPORT_DTYPE = false;
                    u_assert(NON_SUPPORT_DTYPE, "using tensor declared without given type parameter [template T]?");
                }
            }

            template<class C, typename T, template<typename, typename > class Fun, class ...Args>
            void run2_(C &dst, const C &src, Args &&...args) {
                switch (src.type()) {
                    case u::tensor::DType::int8: Fun<T, char>::run(dst.ref(), src.cref(), dst.shape(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint8: Fun<T, unsigned char>::run(dst.ref(), src.cref(), dst.shape(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::int16: Fun<T, short>::run(dst.ref(), src.cref(), dst.shape(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint16: Fun<T, unsigned short>::run(dst.ref(), src.cref(), dst.shape(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::int32: Fun<T, int>::run(dst.ref(), src.cref(), dst.shape(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint32: Fun<T, unsigned int>::run(dst.ref(), src.cref(), dst.shape(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::int64: Fun<T, long>::run(dst.ref(), src.cref(), dst.shape(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint64: Fun<T, unsigned long>::run(dst.ref(), src.cref(), dst.shape(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::float32: Fun<T, float>::run(dst.ref(), src.cref(), dst.shape(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::float64: Fun<T, double>::run(dst.ref(), src.cref(), dst.shape(), src.shape(), std::forward<Args>(args)...); break;
                    default:
                    bool NON_SUPPORT_DTYPE = false;
                    u_assert(NON_SUPPORT_DTYPE, "using tensor declared without given type parameter [template T2]?");
                }
            }

            template<class C, template<typename, typename > class Fun, class ...Args>
            void run2(C &dst, const C &src, Args &&...args) {
                switch (dst.type()) {
                    case u::tensor::DType::int8: run2_<C, char, Fun, Args...>(dst, src, std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint8:	run2_<C, unsigned char, Fun, Args...>(dst, src, std::forward<Args>(args)...); break;
                    case u::tensor::DType::int16: run2_<C, short, Fun, Args...>(dst, src, std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint16: run2_<C, unsigned short, Fun, Args...>(dst, src, std::forward<Args>(args)...); break;
                    case u::tensor::DType::int32: run2_<C, int, Fun, Args...>(dst, src, std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint32: run2_<C, unsigned int, Fun, Args...>(dst, src, std::forward<Args>(args)...); break;
                    case u::tensor::DType::int64: run2_<C, long, Fun, Args...>(dst, src, std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint64: run2_<C, unsigned long, Fun, Args...>(dst, src, std::forward<Args>(args)...); break;
                    case u::tensor::DType::float32: run2_<C, float, Fun, Args...>(dst, src, std::forward<Args>(args)...); break;
                    case u::tensor::DType::float64: run2_<C, double, Fun, Args...>(dst, src, std::forward<Args>(args)...); break;
                    default:
                    bool NON_SUPPORT_DTYPE = false;
                    u_assert(NON_SUPPORT_DTYPE, "using tensor declared without given type parameter [template T1]?");
                }
            }

            template<class C, typename T1, typename T2, template<typename, typename, typename > class Fun, class ...Args>
            void run3__(C &dst, const C &src1, const C &src2, Args &&...args) {
                switch (src2.type()) {
                    case u::tensor::DType::int8: Fun<T1, T2, char>::run(dst.ref(), src1.cref(), src2.cref(), dst.shape(), src1.shape(), src2.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint8: Fun<T1, T2, unsigned char>::run(dst.ref(), src1.cref(), src2.cref(), dst.shape(), src1.shape(), src2.shape(), std::forward<Args>(args)...);	break;
                    case u::tensor::DType::int16: Fun<T1, T2, short>::run(dst.ref(), src1.cref(), src2.cref(), dst.shape(), src1.shape(), src2.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint16: Fun<T1, T2, unsigned short>::run(dst.ref(), src1.cref(), src2.cref(), dst.shape(), src1.shape(), src2.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::int32: Fun<T1, T2, int>::run(dst.ref(), src1.cref(), src2.cref(), dst.shape(), src1.shape(), src2.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint32: Fun<T1, T2, unsigned int>::run(dst.ref(), src1.cref(), src2.cref(), dst.shape(), src1.shape(), src2.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::int64: Fun<T1, T2, long>::run(dst.ref(), src1.cref(), src2.cref(), dst.shape(), src1.shape(), src2.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint64: Fun<T1, T2, unsigned long>::run(dst.ref(), src1.cref(), src2.cref(), dst.shape(), src1.shape(), src2.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::float32: Fun<T1, T2, float>::run(dst.ref(), src1.cref(), src2.cref(), dst.shape(), src1.shape(), src2.shape(), std::forward<Args>(args)...); break;
                    //case u:tensor::DType::float64： Fun<T1, T2, double>::run(dst.ref(), src1.cref(), src2.cref(), dst.shape(), src1.shape(), src2.shape(), std::forward<Args>(args)...); break;
                    default:
                    bool NON_SUPPORT_DTYPE = false;
                    u_assert(NON_SUPPORT_DTYPE, "using tensor declared without given type in parameter [template 3]?");
                }
            }

            template<class C, typename T, template<typename, typename, typename > class Fun, class ...Args>
            void run3_(C &dst, const C &src1, const C &src2, Args &&...args) {
                switch (src1.type()) {
                    case u::tensor::DType::int8: run3__<C, T, char, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint8: run3__<C, T, unsigned char, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    case u::tensor::DType::int16: run3__<C, T, short, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint16: run3__<C, T, unsigned short, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    case u::tensor::DType::int32: run3__<C, T, int, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint32: run3__<C, T, unsigned int, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    case u::tensor::DType::int64: run3__<C, T, long, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint64: run3__<C, T, unsigned long, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    case u::tensor::DType::float32: run3__<C, T, float, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    case u::tensor::DType::float64: run3__<C, T, double, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    default:
                    bool NON_SUPPORT_DTYPE = false;
                    u_assert(NON_SUPPORT_DTYPE, "declare tensor without given type parameter [template 2]?");
                }
            }

            template<class C, template<typename, typename, typename > class Fun, class ...Args>
            void run3(C &dst, const C &src1, const C &src2, Args &&...args) {
                switch (dst.type()) {
                    case u::tensor::DType::int8: run3_<C, char, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint8: run3_<C, unsigned char, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    case u::tensor::DType::int16: run3_<C, short, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint16: run3_<C, unsigned short, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    case u::tensor::DType::int32: run3_<C, int, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint32: run3_<C, unsigned int, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    case u::tensor::DType::int64: run3_<C, long, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint64: run3_<C, unsigned long, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    case u::tensor::DType::float32: run3_<C, float, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    case u::tensor::DType::float64: run3_<C, double, Fun, Args...>(dst, src1, src2, std::forward<Args>(args)...); break;
                    default:
                    bool NON_SUPPORT_DTYPE = false;
                    u_assert(NON_SUPPORT_DTYPE, "declare tensor without given type parameter [template 1]?");
                }
            }

            template<typename T>
            class Print {
            public:
                // print elements between [begin, end)
                static void run(const unsigned char *dst, const Shape &shape, size_t begin, size_t end, std::ostream &os) {
                    if (shape.rank() == 0) {
                        if (dst != nullptr) {
                            const T * const data_ = reinterpret_cast<const T* const >(dst);
                            os << *data_ << std::flush;
                        } else {
                            os << "null" << std::flush;
                        }
                    } else {
                        u_assert(begin < end, u::format("`begin` must be less than `end` (%zu vs %zu)", begin, end));
                        u_assert(end <= shape.volume(), u::format("`end` overflow (%zu vs %zu)", end, shape.volume()));
                        if (dst != nullptr) {
                            const T * const data_ = reinterpret_cast<const T* const >(dst);
                            for (size_t i = begin; i < end; ++i) {
                                os << data_[i] << std::flush;
                                if (i != end - 1) {
                                    os << " , ";
                                }
                            }
                        } else {
                            for (size_t i = begin; i < end; ++i) {
                                os << "null" << std::flush;
                                if (i != end - 1) {
                                    os << " , ";
                                }
                            }
                        }
                    }
                }
            };

            #ifdef USE_CUDA

            template<typename To, typename T1, typename T2>
            class Equal {
            public:
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    gpu::equal<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class Add {
            public:
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    gpu::add<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class Subtract {
            public:
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    gpu::subtract<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class Multiply {
            public:
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    gpu::multiply<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class Divide {
            public:
                static To run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    gpu::divide<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class Mod {
            public:
                static To run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    gpu::mod<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class Pow {
            public:
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    gpu::pow<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            // natural logarithm
            template<typename To, typename Ti>
            class Log {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    gpu::log<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class Log10 {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    gpu::log10<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class Experiential  {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    gpu::exp<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class Invert {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    cpu::invert<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class Minus {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    gpu::minus<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class Absolute {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(0, 0);
                    gpu::abs<To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class Clip {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, double min, double max) {
                    u_fun_enter(0, 0);
                    gpu::clip<To, Ti>(dst, src, dshape, sshape, min, max);
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class Assign {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    gpu::assign<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class Sum {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(0, 0);
                    gpu::sum<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class Mean {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(0, 0);
                    gpu::mean<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class StdDev {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(0, 0);
                    gpu::stddev<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class Max {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(0, 0);
                    gpu::max<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class Min {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(0, 0);
                    gpu::min<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class ArgMax {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(0, 0);
                    gpu::argmax<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class ArgMin {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(0, 0);
                    gpu::argmin<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class Transpose {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, const std::vector<size_t> &dims, const std::map<size_t, size_t> &dim_map) {
                    u_fun_enter(0, 0);
                    u_assert(dst != nullptr && src != nullptr,"both dst and src must not be null pointer");
                    u_assert(dst != src, "transpose cannot be called with in-place mode");
                    u_assert(dim_map.size() == dshape.size(),u::format("transpose changes size not match (%zu, vs %zu)",dim_map.size(), dshape.size()));
                    u_assert(dshape.rank() == sshape.rank() && dshape.rank() > 0,u::format("dimensions not match (%zu vs %zu)", dshape.rank(),dshape.rank()));
                    const Ti * const src_ = reinterpret_cast<const Ti * const >(src);
                    To * const dst_ = reinterpret_cast<To * const >(dst);
                    std::vector<double> dprod(dshape.rank(), 1);
                    std::vector<double> diprod(dshape.rank(), 1);
                    for (int i = static_cast<int>(dshape.rank()) - 2; i >= 0; --i) {
                        diprod[i] = diprod[i + 1] * dshape[i + 1];
                        dprod[i] = dprod[i + 1] * sshape[i + 1];
                    }
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (size_t index = 0; index < static_cast<size_t>(dprod[0] * sshape[0]); ++index) {
                        size_t idx = index;
                        size_t nidx = 0;
                        for (size_t dim = 0; dim < dprod.size(); ++dim) {
                            size_t i = static_cast<size_t>(idx / dprod[dim]);
                            nidx += i * diprod[dim_map.at(static_cast<unsigned int>(dim))];
                            idx = idx % static_cast<size_t>(dprod[dim]);
                        }
                        dst_[nidx] = src_[index];
                    }
                    u_fun_exit(0, 0);
                }
            };

            #else

            template<typename To, typename T1, typename T2>
            class Equal {
            public:
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    cpu::equal<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class NotEqual {
            public:
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    cpu::nequal<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class Greater {
            public:
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    cpu::greater<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class GreaterEqual {
            public:
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    cpu::gequal<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class Less {
            public:
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    cpu::less<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class LessEqual {
            public:
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    cpu::lequal<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class Add {
            public:
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    cpu::add<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class Subtract {
            public:
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    cpu::subtract<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class Multiply {
            public:
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    cpu::multiply<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class Divide {
            public:
                static To run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    cpu::divide<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class Mod {
            public:
                static To run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    cpu::mod<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class Pow {
            public:
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    cpu::pow<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            // natural logarithm
            template<typename To, typename Ti>
            class Log {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    cpu::log<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class Log10 {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    cpu::log10<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class Experiential  {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    cpu::exp<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class Invert {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    cpu::invert<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class Minus {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    cpu::minus<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class Absolute {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(0, 0);
                    cpu::abs<To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class Clip {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, double min, double max) {
                    u_fun_enter(0, 0);
                    cpu::clip<To, Ti>(dst, src, dshape, sshape, min, max);
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class Assign {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    cpu::assign<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class Sum {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(0, 0);
                    cpu::sum<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class Mean {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(0, 0);
                    cpu::mean<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class StdDev {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(0, 0);
                    cpu::stddev<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class Max {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(0, 0);
                    cpu::max<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class Min {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(0, 0);
                    cpu::min<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class ArgMax {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(0, 0);
                    cpu::argmax<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class ArgMin {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(0, 0);
                    cpu::argmin<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class Transpose {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, const std::vector<size_t> &dims, const std::map<size_t, size_t> &dim_map) {
                    u_fun_enter(0, 0);
                    u_assert(dst != nullptr && src != nullptr,"both dst and src must not be null pointer");
                    u_assert(dst != src, "transpose cannot be called with in-place mode");
                    u_assert(dim_map.size() == dshape.size(),u::format("transpose changes size not match (%zu, vs %zu)",dim_map.size(), dshape.size()));
                    u_assert(dshape.rank() == sshape.rank() && dshape.rank() > 0,u::format("dimensions not match (%zu vs %zu)", dshape.rank(),dshape.rank()));
                    const Ti * const src_ = reinterpret_cast<const Ti * const >(src);
                    To * const dst_ = reinterpret_cast<To * const >(dst);
                    std::vector<double> dprod(dshape.rank(), 1);
                    std::vector<double> diprod(dshape.rank(), 1);
                    for (int i = static_cast<int>(dshape.rank()) - 2; i >= 0; --i) {
                        diprod[i] = diprod[i + 1] * dshape[i + 1];
                        dprod[i] = dprod[i + 1] * sshape[i + 1];
                    }
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (size_t index = 0; index < static_cast<size_t>(dprod[0] * sshape[0]); ++index) {
                        size_t idx = index;
                        size_t nidx = 0;
                        for (size_t dim = 0; dim < dprod.size(); ++dim) {
                            size_t i = static_cast<size_t>(idx / dprod[dim]);
                            nidx += i * diprod[dim_map.at(static_cast<unsigned int>(dim))];
                            idx = idx % static_cast<size_t>(dprod[dim]);
                        }
                        dst_[nidx] = src_[index];
                    }
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class Any {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, bool positive) {
                    u_fun_enter(0, 0);
                    u_assert(dst != nullptr && src != nullptr,"both dst and src must not be null pointer");
                    u_assert(dst != src, "any cannot be called with in-place mode");
                    u_assert(dshape.volume() == 1, u::format("destination of tensor of `Any` operation must have volume of 1 (scalar). given %zu", dshape.volume()));
                    const Ti * const src_ = reinterpret_cast<const Ti * const >(src);
                    To * const dst_ = reinterpret_cast<To * const >(dst);
                    size_t svolume = sshape.volume();
                    (*dst_) = static_cast<unsigned char>(!positive);
                    if (positive) {
                        for (size_t i=0; i<svolume; ++i) {
                            if (src_[i]) {
                                (*dst_) = 1;
                                break;
                            }
                        }
                    } else {
                        for (size_t i=0; i<svolume; ++i) {
                            if (! src_[i]) {
                                (*dst_) = 1;
                                break;
                            }
                        }
                    }
                    u_fun_exit(0, 0);
                }
            };

            template<typename To, typename Ti>
            class All {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, bool positive) {
                    u_fun_enter(0, 0);
                    u_fun_enter(0, 0);
                    u_assert(dst != nullptr && src != nullptr,"both dst and src must not be null pointer");
                    u_assert(dst != src, "any cannot be called with in-place mode");
                    u_assert(dshape.volume() == 1, u::format("destination of tensor of `All` operation must have volume of 1 (scalar). given %zu", dshape.volume()));
                    const Ti * const src_ = reinterpret_cast<const Ti * const >(src);
                    To * const dst_ = reinterpret_cast<To * const >(dst);
                    size_t svolume = sshape.volume();
                    (*dst_) = static_cast<unsigned char>(positive);
                    if (positive) {
                        for (size_t i=0; i<svolume; ++i) {
                            if (!src_[i]) {
                                (*dst_) = 0;
                                break;
                            }
                        }
                    } else {
                        for (size_t i=0; i<svolume; ++i) {
                            if (src_[i]) {
                                (*dst_) = 0;
                                break;
                            }
                        }
                    }
                    u_fun_exit(0, 0);
                }
            };
            #endif
        }
    }
}

#endif
