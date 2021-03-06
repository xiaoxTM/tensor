#ifndef __U_TENSOR_OPERATION_CPU_HPP__
#define __U_TENSOR_OPERATION_CPU_HPP__

/***
u-op-cpu.hpp base functions for tensor
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

#include <cmath>
#include <vector>
#include <map>
#include <limits>
#include <iomanip>

#include <libu/u-log>

#include "../u-shape.hpp"
#include "../u-dtype.hpp"
#include "../u-mm.hpp"

namespace u {
    namespace tensor {
        namespace op {
            // include operation classes
            #include "./imp/cpu/cop"
            namespace cpu {

                /** broadcast tensor with shape `sshape` to tensor with shape `dshape`
                 ** broadcast divides sshape into pieces from axis (beg_axis) where
                 ** `dshape` and `sshape` have same size, to axis (end_axis) where
                 ** `dshape` and `sshape` have different size.
                 **  e.g.: dshape = [3,2,2,4,2,2,5], sshape = [3,1,1,4,1,1,5]
                 **  will be divided to [end_axis, beg_axis]:
                 **  [ [5, 3], [2, 0] ]
                 **
                 **  then, for each segment, do broadcast
                 **  for (i=beg_axis, i>end_axis; --i) {
                 **      broadcast from i to end_axis
                 **  }
                */
                template<typename To, typename Ti>
                void broadcast(unsigned char *dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    u_assert(dshape.rank() == sshape.rank(), u::format("broadcast two tensor should have same rank. given (%zu vs %zu)", dshape.rank(), sshape.rank()));
                    To *dst_ = reinterpret_cast<To *>(dst);
                    const Ti * const src_ = reinterpret_cast<const Ti * const>(src);
                    if (dshape == sshape) { // just in case
                        std::copy_n(src_, dshape.volume(), dst_);
                    } else {
                        int beg_axis = dshape.next_axis(static_cast<int>(dshape.rank())-1, sshape, false);
                        // initialize dst with src
                        size_t times = sshape[beg_axis];
                        if (beg_axis != 0) {
                            times = sshape.volume(0, beg_axis, true, false);
                        }
                        size_t intra_strides = 1;
                        if (beg_axis != static_cast<int>(sshape.rank())-1) {
                            intra_strides = sshape.volume(beg_axis, -1);
                        }


                        #ifdef _OPENMP
                        #pragma omp parallel for
                        #endif
                        for (size_t t=0; t<times; ++t) {
                            size_t begin = sshape.offsetmap(dshape, t, beg_axis) * intra_strides;
                            std::copy_n(src_ + t * intra_strides, intra_strides, dst_ + begin);
                        }

                        int end_axis = dshape.next_axis(static_cast<int>(beg_axis), sshape);

                        while (true) {
                            for (int i = beg_axis; i > end_axis; -- i) {
                                times = dshape[i];
                                if (i != 0) {
                                    times = sshape.volume(0, i, true, false);
                                }
                                intra_strides = 1;
                                if (i != static_cast<int>(dshape.rank())-1) {
                                    intra_strides = dshape.volume(i, -1, false);
                                }

                                size_t repeats = static_cast<int>(dshape[i] / sshape[i]);

                                #ifdef _OPENMP
                                #pragma omp parallel for
                                #endif
                                for (size_t t=0; t<times; ++t) {
                                    size_t begin = sshape.offsetmap(dshape, t, i) * intra_strides;
                                    for (size_t r=1; r<repeats; ++r) {
                                        std::copy_n(dst_ + begin, intra_strides, dst_ + begin + r * intra_strides);
                                    }
                                }
                            }
                            beg_axis = dshape.next_axis(end_axis, sshape, false);
                            if (beg_axis < 0) {
                                break;
                            }
                            end_axis = dshape.next_axis(beg_axis, sshape);
                        }
                    }
                    u_fun_exit(0, -2);
                }

                template<typename T>
                void concatenate(unsigned char * dst, const std::vector<const unsigned char *> &srcs, const Shape &shape, const std::vector<Shape> &shapes, int axis) {
                    u_fun_enter(2, 0);
                    T *dst_ = reinterpret_cast<T*>(dst);
                    size_t naxis = shape.axis_normalize(axis);
                    std::vector<size_t> offsets(shapes.size());
                    for (size_t s=0; s<srcs.size(); ++s) {
                        offsets[s] = shapes[s].volume(naxis, -1);
                    }
                    size_t offset = shape.volume(naxis, -1);
                    size_t times = 1;
                    if (naxis != 0) {
                        times = shape.volume(0, naxis, true, false);
                    }
                    for (size_t t=0; t<times; ++t) {
                        size_t begin = t * offset;
                        for (size_t s=0; s<srcs.size(); ++s) {
                            std::copy_n(reinterpret_cast<const T*>(srcs[s]) + (t * offsets[s]), offsets[s], dst_ + begin);
                            begin += offsets[s];
                        }
                    }
                    u_fun_exit(0, -2);
                }

                template<typename T>
                void split(std::vector<unsigned char *> &dsts, const unsigned char *src, const std::vector<Shape> &shapes, const Shape &shape, size_t axis) {
                    u_fun_enter(2, 0);
                    const T *src_ = reinterpret_cast<const T*>(src);
                    std::vector<size_t> offsets(shapes.size());
                    for (size_t s=0; s<dsts.size(); ++s) {
                        offsets[s] = shapes[s].volume(axis, -1);
                    }
                    size_t offset = shape.volume(axis, -1);
                    size_t times = 1;
                    if (axis != 0) {
                        times = shape.volume(0, axis, true, false);
                    }
                    for (size_t t=0; t<times; ++t) {
                        size_t begin = t * offset;
                        for (size_t s=0; s<dsts.size(); ++s) {
                            std::copy_n(src_ + begin, offsets[s], reinterpret_cast<T*>(dsts[s]) + (t * offsets[s]));
                            begin += offsets[s];
                        }
                    }
                    u_fun_exit(0, -2);
                }

                template<template<typename, typename > class ops, typename To, typename Ti, class ...Args>
                void unitary_op_run(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape, Args &&...args) {
                    // element wise operation
                    u_fun_enter(2, 0);
                    u_assert(dst != nullptr && src != nullptr, "both dst and src must not be null pointer");
                    size_t dvolume = dshape.volume();
                    size_t svolume = sshape.volume();
                    u_assert(dvolume > 0 && svolume > 0, u::format("dsize and ssize must be greater than zero (%zu vs %zu)", dvolume, svolume));
                    u_assert(dshape == sshape, u::format("dshape and sshape must be the same (%s vs %s)", dshape.c_str(), sshape.c_str()));
                    const Ti *_src_ = reinterpret_cast<const Ti *>(src);
                    To *_dst_ = reinterpret_cast<To *>(dst);
                    ops<To, Ti>::run(_dst_, _src_, dvolume, std::forward<Args>(args)...);
                    u_fun_exit(0, -2);
                }

                template<template<typename, typename, typename > class ops, typename To, typename T1, typename T2, class ...Args>
                void binary_op_run(unsigned char * const dst, const unsigned char * const src1, const unsigned char * const src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2, Args &&...args) {
                    u_fun_enter(2, 0);
                    assert(dst != nullptr && src1 != nullptr && src2 != nullptr);
                    auto shapes = Shape::adapt_shape(sshape1, sshape2);
                    u_assert(std::get<0>(shapes) == dshape, u::format("dshape [%s] not match broadcasted shape [%s]", dshape.c_str(), std::get<0>(shapes).c_str()));
                    Shape _sshape1_ = std::get<1>(shapes);
                    Shape _sshape2_ = std::get<2>(shapes);
                    std::shared_ptr<unsigned char> broadcasted_src1;
                    std::shared_ptr<unsigned char> broadcasted_src2;
                    #ifdef _OPENMP
                    #pragma omp parallel sections {
                    #pragma omp seciton {
                    #endif
                    if (dshape == sshape1) {
                        broadcasted_src1.reset(const_cast<unsigned char*>(src1), mm::no_free);
                    } else {
                        broadcasted_src1.reset(mm::malloc(dshape.volume(), sizeof(T1), 0), mm::mfree);
                        broadcast<T1, T1>(broadcasted_src1.get(), src1, dshape, _sshape1_);
                    }
                    #ifdef _OPENMP
                    }
                    #pragma omp section {
                    #endif
                    if (dshape == sshape2) {
                        broadcasted_src2.reset(const_cast<unsigned char*>(src2), mm::no_free);
                    } else {
                        broadcasted_src2.reset(mm::malloc(dshape.volume(), sizeof(T2), 0), mm::mfree);
                        broadcast<T2, T2>(broadcasted_src2.get(), src2, dshape, _sshape2_);
                    }
                    #ifdef _OPENMP
                    }
                    }
                    #endif
                    ops<To, T1, T2>::run(reinterpret_cast<To*>(dst), reinterpret_cast<const T1* const>(broadcasted_src1.get()), reinterpret_cast<const T2* const>(broadcasted_src2.get()), dshape.volume());
                    u_fun_exit(0, -2);
                }

                template<template<typename, typename > class ops, typename To, typename Ti, class ...Args>
                void dimension_op_run(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape, int axis, Args &&...args) {
                    // dimension operations
                    // c++ stores tensors in the form of array.
                    // for example, we have 3D array with shape of {2, 3, 8}
                    //[[[0 , 1 , 2 , 3 , 4 , 5 , 6 , 7],
                    //  [8 , 9 , 10 , 11 , 12 , 13 , 14 , 15],
                    //  [16 , 17 , 18 , 19 , 20 , 21 , 22 , 23]],
                    // [[24 , 25 , 26 , 27 , 28 , 29 , 30 , 31],
                    //  [32 , 33 , 34 , 35 , 36 , 37 , 38 , 39],
                    //  [40 , 41 , 42 , 43 , 44 , 45 , 46 , 47]]]
                    // is stored as [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ..., 47] vector
                    // assume we want to do operation on axis = 1, that is:
                    // dim_size = 3 = {2, 3, 8}[1]
                    // previous = prod(shape[:axis]) = prod(2)st, src1
                    // later = prod(shape[axis:]) = prod(3 * 8)
                    // offset = prod(shape[axis+1]) = prod(8)
                    // consider the 3D array as three parts:
                    // previous / later / offset
                    // NOTE that the result tensor in vector form have size of
                    // previous x offset
                    // therefore two layers loops cover all element of the result tensor
                    // for prev in range(previous):
                    //    for ofs in range(offset):
                    //        result(prev, ofs) = op([for dim in range(dim_size)])
                    // operation is done in dim_size damon (which is a OP class)
                    u_fun_enter(2, 0);
                    u_assert(dst != nullptr && src != nullptr, "both dst and src must not be null pointer");
                    const Ti * const src_ = reinterpret_cast<const Ti * const >(src);
                    To * const dst_ = reinterpret_cast<To * const >(dst);
                    #ifdef _OPENMP
                    #pragma omp paralle for
                    #endif
                    std::tuple<size_t, size_t, size_t, size_t> split = sshape.split(axis);
                    size_t dim_size = std::get<0>(split);
                    size_t previous = std::get<1>(split);
                    size_t later    = std::get<2>(split);
                    size_t offset   = std::get<3>(split);
                    for (size_t prev = 0; prev < previous; ++prev) {
                        size_t outer_offset = prev * offset;
                        for (size_t inner = 0; inner < offset; ++inner) {
                            dst_[outer_offset + inner] = ops<To, Ti>::run(src_, dim_size, prev, later, inner, offset, std::forward<Args>(args)...);
                        }
                    }
                    u_fun_exit(0, -2);
                }

                template<typename To, typename T1, typename T2>
                inline void add(unsigned char * const dst, const unsigned char * const src1, const unsigned char * const src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    u_fun_enter(2, 0);
                    binary_op_run<aop::Add, To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename T1, typename T2>
                inline void subtract(unsigned char * const dst, const unsigned char * const src1, const unsigned char * const src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    u_fun_enter(2, 0);
                    binary_op_run<aop::Subtract, To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename T1, typename T2>
                inline void multiply(unsigned char * const dst, const unsigned char * const src1, const unsigned char * const src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    u_fun_enter(2, 0);
                    binary_op_run<aop::Multiply, To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename T1, typename T2>
                inline void divide(unsigned char * const dst, const unsigned char * const src1, const unsigned char * const src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    u_fun_enter(2, 0);
                    binary_op_run<aop::Divide, To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename T1, typename T2>
                inline void pow(unsigned char * const dst, const unsigned char * const src1, const unsigned char * const src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    u_fun_enter(2, 0);
                    binary_op_run<aop::Pow, To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename T1, typename T2>
                inline void max(unsigned char * const dst, const unsigned char * const src1, const unsigned char * const src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    u_fun_enter(2, 0);
                    binary_op_run<aop::Maximum, To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename T1, typename T2>
                inline void min(unsigned char * const dst, const unsigned char * const src1, const unsigned char * const src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    u_fun_enter(2, 0);
                    binary_op_run<aop::Minimum, To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename T1, typename T2>
                inline void mod(unsigned char * const dst, const unsigned char * const src1, const unsigned char * const src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    u_fun_enter(2, 0);
                    binary_op_run<aop::Mod, To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename T1, typename T2>
                inline void equal(unsigned char * const dst, const unsigned char * const src1, const unsigned char * const src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    u_fun_enter(2, 0);
                    binary_op_run<aop::Equal, To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename T1, typename T2>
                inline void nequal(unsigned char * const dst, const unsigned char * const src1, const unsigned char * const src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    u_fun_enter(2, 0);
                    binary_op_run<aop::NotEqual, To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename T1, typename T2>
                inline void greater(unsigned char * const dst, const unsigned char * const src1, const unsigned char * const src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    u_fun_enter(2, 0);
                    binary_op_run<aop::Greater, To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename T1, typename T2>
                inline void gequal(unsigned char * const dst, const unsigned char * const src1, const unsigned char * const src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    u_fun_enter(2, 0);
                    binary_op_run<aop::GreaterEqual, To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename T1, typename T2>
                inline void less(unsigned char * const dst, const unsigned char * const src1, const unsigned char * const src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    u_fun_enter(2, 0);
                    binary_op_run<aop::Less, To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename T1, typename T2>
                inline void lequal(unsigned char * const dst, const unsigned char * const src1, const unsigned char * const src2, const Shape &dshape, const Shape &sshape1, const Shape &sshape2) {
                    u_fun_enter(2, 0);
                    binary_op_run<aop::LessEqual, To, T1, T2>(dst, src1, src2, dshape, sshape1, sshape2);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void log(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::Log, To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void log10(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::Log10, To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void sqrt(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::SquareRoot, To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void sin(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::Sine, To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void cos(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::Cosine, To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void tan(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::Tangent, To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void tanh(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::HyperbolicTangent, To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void round(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::Round, To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void ceil(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::Ceil, To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void floor(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::Floor, To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void isinf(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::IsInfinite, To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void isnan(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::IsNaN, To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void isfinite(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::IsFinite, To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void exp(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::Experiential, To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void invert(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::Invert, To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void minus(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::Minus, To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void clip(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape, double min, double max) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::Clip, To, Ti>(dst, src, dshape, sshape, min, max);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void abs(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::Absolute, To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void assign(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape) {
                    u_fun_enter(2, 0);
                    unitary_op_run<aop::Assign, To, Ti>(dst, src, dshape, sshape);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void sum(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(2, 0);
                    dimension_op_run<aop::Sum, To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void mean(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(2, 0);
                    dimension_op_run<aop::Mean, To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void stddev(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(2, 0);
                    dimension_op_run<aop::StdDev, To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void max(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(2, 0);
                    dimension_op_run<aop::Max, To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void min(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(2, 0);
                    dimension_op_run<aop::Min, To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void argmax(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(2, 0);
                    dimension_op_run<aop::ArgMax, To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, -2);
                }

                template<typename To, typename Ti>
                inline void argmin(unsigned char * const dst, const unsigned char * const src, const Shape &dshape, const Shape &sshape, int axis) {
                    u_fun_enter(2, 0);
                    dimension_op_run<aop::ArgMin, To, Ti>(dst, src, dshape, sshape, axis);
                    u_fun_exit(0, -2);
                }
            }
        }
    }
}

#endif
