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
            #include "./imp/op/cpu/cop"
            #endif
        }
    }
}

#endif
