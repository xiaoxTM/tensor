#ifndef __U_TENSOR_OPERATION_HPP__
#define __U_TENSOR_OPERATION_HPP__

/***
u-op.hpp base functions for tensor
Copyright (C) 2017  Renweu Gao

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
***/

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

            /**
             ** run multiple tensors in term of vector<Tensor>
             ** NOTE all tensors should have same datatype
            */
            template<class C, template<typename > class Fun, class ...Args>
            void runv(std::vector<C> &dst, const C &src, Args &&...args) {
                std::vector<unsigned char*> data;
                for (size_t i=0; i<dst.size(); ++i) {
                    data.push_back(dst[i].ref());
                }
                switch (src.type()) {
                    case u::tensor::DType::int8: Fun<char>::run(data, src.cref(), src.shape(), std::forward<Args>(args)...);break;
                    case u::tensor::DType::uint8: Fun<unsigned char>::run(data, src.cref(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::int16: Fun<short>::run(data, src.cref(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint16: Fun<unsigned short>::run(data, src.cref(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::int32: Fun<int>::run(data, src.cref(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint32: Fun<unsigned int>::run(data, src.cref(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::int64: Fun<long>::run(data, src.cref(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint64: Fun<unsigned long>::run(data, src.cref(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::float32: Fun<float>::run(data, src.cref(), src.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::float64: Fun<double>::run(data, src.cref(), src.shape(), std::forward<Args>(args)...); break;
                    default:
                    bool NON_SUPPORT_DTYPE = false;
                    u_assert(NON_SUPPORT_DTYPE, u::format("using tensor declared without given type parameter (%s) [template T]?", dtype_str(src.type()).c_str()));
                }
            }

            /**
             ** run multiple tensors in term of vector<Tensor>
             ** NOTE all tensors should have same datatype
            */
            template<class C, template<typename > class Fun, class ...Args>
            void runv(C & dst, const std::vector<C> &srcs, Args &&...args) {
                std::vector<const unsigned char*> data;
                std::vector<Shape> shapes;
                for (size_t i=0; i<srcs.size(); ++i) {
                    data.push_back(srcs[i].ref());
                    shapes.push_back(srcs[i].shape());
                }
                switch (dst.type()) {
                    case u::tensor::DType::int8: Fun<char>::run(dst.ref(), data, dst.shape(), shapes, std::forward<Args>(args)...);break;
                    case u::tensor::DType::uint8: Fun<unsigned char>::run(dst.ref(), data, dst.shape(), shapes, std::forward<Args>(args)...); break;
                    case u::tensor::DType::int16: Fun<short>::run(dst.ref(), data, dst.shape(), shapes, std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint16: Fun<unsigned short>::run(dst.ref(), data, dst.shape(), shapes, std::forward<Args>(args)...); break;
                    case u::tensor::DType::int32: Fun<int>::run(dst.ref(), data, dst.shape(), shapes, std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint32: Fun<unsigned int>::run(dst.ref(), data, dst.shape(), shapes, std::forward<Args>(args)...); break;
                    case u::tensor::DType::int64: Fun<long>::run(dst.ref(), data, dst.shape(), shapes, std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint64: Fun<unsigned long>::run(dst.ref(), data, dst.shape(), shapes, std::forward<Args>(args)...); break;
                    case u::tensor::DType::float32: Fun<float>::run(dst.ref(), data, dst.shape(), shapes, std::forward<Args>(args)...); break;
                    case u::tensor::DType::float64: Fun<double>::run(dst.ref(), data, dst.shape(), shapes, std::forward<Args>(args)...); break;
                    default:
                    bool NON_SUPPORT_DTYPE = false;
                    u_assert(NON_SUPPORT_DTYPE, u::format("using tensor declared without given type parameter (%s) [template T]?", dtype_str(dst.type()).c_str()));
                }
            }

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
                    u_assert(NON_SUPPORT_DTYPE, u::format("using tensor declared without given type parameter (%s) [template T]?", dtype_str(src.type()).c_str()));
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
                    u_assert(NON_SUPPORT_DTYPE, u::format("using tensor declared without given type parameter (%s) [template T2]?", dtype_str(src.type()).c_str()));
                }
            }

            template<class C, template<typename, typename > class Fun, class ...Args>
            void run2(C &dst, const C &src, Args &&...args) {
                switch (dst.type()) {
                    case u::tensor::DType::int8: run2_<C, char, Fun, Args...>(dst, src, std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint8:    run2_<C, unsigned char, Fun, Args...>(dst, src, std::forward<Args>(args)...); break;
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
                    u_assert(NON_SUPPORT_DTYPE, u::format("using tensor declared without given type parameter (%s) [template T1]?", dtype_str(dst.type()).c_str()));
                }
            }

            template<class C, typename T1, typename T2, template<typename, typename, typename > class Fun, class ...Args>
            void run3__(C &dst, const C &src1, const C &src2, Args &&...args) {
                switch (src2.type()) {
                    case u::tensor::DType::int8: Fun<T1, T2, char>::run(dst.ref(), src1.cref(), src2.cref(), dst.shape(), src1.shape(), src2.shape(), std::forward<Args>(args)...); break;
                    case u::tensor::DType::uint8: Fun<T1, T2, unsigned char>::run(dst.ref(), src1.cref(), src2.cref(), dst.shape(), src1.shape(), src2.shape(), std::forward<Args>(args)...);    break;
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
                    u_assert(NON_SUPPORT_DTYPE, u::format("using tensor declared without given type in parameter (%s) [template 3]?", dtype_str(src2.type()).c_str()));
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
                    u_assert(NON_SUPPORT_DTYPE, u::format("declare tensor without given type parameter (%s) [template 2]?", dtype_str(src1.type()).c_str()));
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
                    u_assert(NON_SUPPORT_DTYPE, u::format("declare tensor without given type parameter (%s) [template 1]?", dtype_str(dst.type()).c_str()));
                }
            }

            template<typename T>
            class Print {
            public:
                // print elements between [begin, end)
                static void run(const unsigned char *dst, const Shape &shape, size_t begin, size_t end, std::ostream &os, int precision, int width) {
                    if (shape.rank() == 0) {
                        if (dst != nullptr) {
                            const T * const data_ = reinterpret_cast<const T* const >(dst);
                            os << std::setprecision(precision) << std::setw(width) << std::right << *data_ << std::flush;
                        } else {
                            os << std::setprecision(precision) << std::setw(width) << std::right << "null" << std::flush;
                        }
                    } else {
                        u_assert(begin < end, u::format("`begin` must be less than `end` (%zu vs %zu)", begin, end));
                        u_assert(end <= shape.volume(), u::format("`end` overflow (%zu vs %zu)", end, shape.volume()));
                        if (dst != nullptr) {
                            const T * const data_ = reinterpret_cast<const T* const >(dst);
                            for (size_t i = begin; i < end; ++i) {
                                os << std::setprecision(precision) << std::setw(width) << std::right << data_[i] << std::flush;
                                if (i != end - 1) {
                                    os << ", ";
                                }
                            }
                        } else {
                            for (size_t i = begin; i < end; ++i) {
                                os << std::setprecision(precision) << "null" << std::flush;
                                if (i != end - 1) {
                                    os << " , ";
                                }
                            }
                        }
                    }
                }
            };

            #ifdef USE_CUDA

            #else

            template<typename To, typename Ti>
            class Broadcast {
            public:
                static void run(unsigned char *dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    cpu::broadcast<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename T>
            class Concatenate {
            public:
                static void run(unsigned char *dst, const std::vector<const unsigned char *> &srcs, const Shape &shape, const std::vector<Shape> &shapes, int axis) {
                    cpu::concatenate<T>(dst, srcs, shape, shapes, axis);
                }
            };

            template<typename T>
            class Split {
            public:
                static void run(std::vector<unsigned char *> &dsts, const unsigned char * src, const Shape &shape, const std::vector<Shape> &shapes, size_t axis) {
                    cpu::split<T>(dsts, src, shapes, shape, axis);
                }
            };

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
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    cpu::divide<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class Mod {
            public:
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
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

            template<typename To, typename T1, typename T2>
            class Maximum {
            public:
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    cpu::max<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                }
            };

            template<typename To, typename T1, typename T2>
            class Minimum {
            public:
                static void run(unsigned char *dst, const unsigned char *src1, const unsigned char *src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    cpu::min<To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
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
            class SquareRoot {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    cpu::sqrt<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class Sine {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    cpu::sin<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class Cosine {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    cpu::cos<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class Tangent {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    cpu::tan<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class HyperbolicTangent {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    cpu::tanh<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class Round {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    cpu::round<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class Ceil {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    cpu::ceil<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class Floor {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    cpu::floor<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class IsInfinite {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    cpu::isinf<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class IsNaN {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    cpu::isnan<To, Ti>(dst, src, dshape, sshape);
                }
            };

            template<typename To, typename Ti>
            class IsFinite {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape) {
                    cpu::isfinite<To, Ti>(dst, src, dshape, sshape);
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
                    u_fun_enter(2, 0);
                    cpu::abs<To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }
            };

            template<typename To, typename Ti>
            class Clip {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, double min, double max) {
                    u_fun_enter(2, 0);
                    cpu::clip<To, Ti>(dst, src, dshape, sshape, min, max);
                    u_fun_exit(0, -2);
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
                    u_fun_enter(2, 0);
                    cpu::sum<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, -2);
                }
            };

            template<typename To, typename Ti>
            class Mean {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(2, 0);
                    cpu::mean<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, -2);
                }
            };

            template<typename To, typename Ti>
            class StdDev {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(2, 0);
                    cpu::stddev<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, -2);
                }
            };

            template<typename To, typename Ti>
            class Max {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(2, 0);
                    cpu::max<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, -2);
                }
            };

            template<typename To, typename Ti>
            class Min {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(2, 0);
                    cpu::min<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, -2);
                }
            };

            template<typename To, typename Ti>
            class ArgMax {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(2, 0);
                    cpu::argmax<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, -2);
                }
            };

            template<typename To, typename Ti>
            class ArgMin {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(2, 0);
                    cpu::argmin<To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, -2);
                }
            };

            template<typename To, typename Ti>
            class Transpose {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, const std::vector<size_t> &dims, const std::map<size_t, size_t> &dim_map) {
                    u_fun_enter(2, 0);
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
                    u_fun_exit(0, -2);
                }
            };

            template<typename To, typename Ti>
            class Any {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, bool positive) {
                    u_fun_enter(2, 0);
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
                    u_fun_exit(0, -2);
                }
            };

            template<typename To, typename Ti>
            class All {
            public:
                static void run(unsigned char *dst, const unsigned char *src, const Shape &dshape, const Shape &sshape, bool positive) {
                    u_fun_enter(2, 0);
                    u_fun_enter(2, 0);
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
                    u_fun_exit(0, -2);
                }
            };
            #endif
        }
    }
}

#endif
