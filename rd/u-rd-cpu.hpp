#ifndef __U_TENSOR_RANDOM_CPU_HPP__
#define __U_TENSOR_RANDOM_CPU_HPP__

/***
u-rd-cpu.hpp base functions for tensor
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

#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace u {

    namespace tensor {

        namespace rd {
            static std::default_random_engine generator_;

            namespace cpu {

                template<typename T>
                inline void bernoulli(unsigned char * const dst, const Shape &shape, double p) {
                    //size_t code = typeid(T).hash_code();
                    std::bernoulli_distribution distribution(p);
                    T * const dst_ = reinterpret_cast<T * const >(dst);
                    size_t size = shape.volume();
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (size_t i = 0; i < size; ++i) {
                        dst_[i] = static_cast<T>(distribution(rd::generator_));
                    }
                }

                template <typename T, typename Tt>
                inline void binomial(unsigned char * const dst, const Shape &shape, Tt top, double p) {
                    std::binomial_distribution <Tt> distribution(top, p);
                    T * const dst_ = reinterpret_cast<T * const >(dst);
                    size_t size = shape.volume();
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (size_t i = 0; i < size; ++i) {
                        dst_[i] = static_cast<T>(distribution(rd::generator_));
                    }
                }

                template<typename T, typename Td>
                inline void cauchy(unsigned char * const dst, const Shape &shape, Td loc, Td scale) {
                    std::cauchy_distribution <Td> distribution(loc, scale);
                    T * const dst_ = reinterpret_cast<T * const >(dst);
                    size_t size = shape.volume();
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (size_t i = 0; i < size; ++i) {
                        dst_[i] = static_cast<T>(distribution(rd::generator_));
                    }
                }

                template<typename T, typename Td>
                inline void chi_squared(unsigned char * const dst, const Shape &shape, Td free_degree) {
                    std::chi_squared_distribution <Td> distribution(free_degree);
                    T * const dst_ = reinterpret_cast<T* const >(dst);
                    size_t size = shape.volume();
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (size_t i = 0; i < size; ++i) {
                        dst_[i] = static_cast<T>(distribution(rd::generator_));
                    }
                }

                template<typename T>
                inline void discrete(unsigned char * const dst, const Shape &shape, size_t nw, double xmin, double xmax) {
                    std::discrete_distribution<T> distribution(nw, xmin, xmax);
                    T * const dst_ = reinterpret_cast<T * const >(dst);
                    size_t size = shape.volume();
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (size_t i = 0; i < size; ++i) {
                        dst_[i] = static_cast<T>(distribution(rd::generator_));
                    }
                }

                template<typename T, typename Td>
                inline void exponential(unsigned char * const dst, const Shape &shape, Td lambda) {
                    std::exponential_distribution < Td > distribution(lambda);
                    T * const dst_ = reinterpret_cast<T * const >(dst);
                    size_t size = shape.volume();
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (size_t i = 0; i < size; ++i) {
                        dst_[i] = static_cast<T>(distribution(rd::generator_));
                    }
                }

                template<typename T, typename Td>
                inline void gamma(unsigned char * const dst, const Shape &shape, Td alpha, Td belta) {
                    std::gamma_distribution < Td > distribution(alpha, belta);
                    T * const dst_ = reinterpret_cast<T * const >(dst);
                    size_t size = shape.volume();
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (size_t i = 0; i < size; ++i) {
                        dst_[i] = static_cast<T>(distribution(rd::generator_));
                    }
                }

                template<typename T, typename Td>
                inline void geometric(unsigned char * const dst, const Shape &shape, double p) {
                    std::geometric_distribution<Td> distribution(p);
                    T * const dst_ = reinterpret_cast<T * const >(dst);
                    size_t size = shape.volume();
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (size_t i = 0; i < size; ++i) {
                        dst_[i] = static_cast<T>(distribution(rd::generator_));
                    }
                }

                template<typename T, typename Td>
                inline void lognormal(unsigned char * const dst, const Shape &shape, Td mean, Td stddev) {
                    std::lognormal_distribution < Td > distribution(mean, stddev);
                    T * const dst_ = reinterpret_cast<T * const >(dst);
                    size_t size = shape.volume();
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (size_t i = 0; i < size; ++i) {
                        dst_[i] = static_cast<T>(distribution(rd::generator_));
                    }
                }

                template<typename T, typename Td>
                inline void normal(unsigned char * const dst, const Shape &shape, Td mean, Td stddev) {
                    std::normal_distribution < Td > distribution(mean, stddev);
                    T * const dst_ = reinterpret_cast<T * const >(dst);
                    size_t size = shape.volume();
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (size_t i = 0; i < size; ++i) {
                        dst_[i] = static_cast<T>(distribution(rd::generator_));
                    }
                }

                template<typename T, typename Td>
                inline void poisson(unsigned char * const dst, const Shape &shape, Td mean) {
                    std::poisson_distribution < Td > distribution(mean);
                    T * const dst_ = reinterpret_cast<T * const >(dst);
                    size_t size = shape.volume();
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (size_t i = 0; i < size; ++i) {
                        dst_[i] = static_cast<T>(distribution(rd::generator_));
                    }
                }

                template<typename T, typename Td>
                inline void student_t(unsigned char * const dst, const Shape &shape, Td free_degree) {
                    std::student_t_distribution < Td > distribution(free_degree);
                    T * const dst_ = reinterpret_cast<T * const >(dst);
                    size_t size = shape.volume();
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (size_t i = 0; i < size; ++i) {
                        dst_[i] = static_cast<T>(distribution(rd::generator_));
                    }
                }

                template<typename T, typename Td>
                inline void weibull(unsigned char * const dst, const Shape &shape, Td a, Td b) {
                    std::weibull_distribution < Td > distribution(a, b);
                    T * const dst_ = reinterpret_cast<T * const >(dst);
                    size_t size = shape.volume();
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (size_t i = 0; i < size; ++i) {
                        dst_[i] = static_cast<T>(distribution(rd::generator_));
                    }
                }

                template<typename T, typename Td>
                inline void uniform_int(unsigned char * const dst, const Shape &shape, Td lower, Td upper) {
                    std::uniform_int_distribution < Td > distribution(lower, upper);
                    T * const dst_ = reinterpret_cast<T * const >(dst);
                    size_t size = shape.volume();
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (size_t i = 0; i < size; ++i) {
                        dst_[i] = static_cast<T>(distribution(rd::generator_));
                    }
                }

                template<typename T, typename Td>
                inline void uniform_real(unsigned char * const dst, const Shape &shape, Td lower, Td upper) {
                    std::uniform_real_distribution < Td > distribution(lower, upper);
                    T * const dst_ = reinterpret_cast<T * const >(dst);
                    size_t size = shape.volume();
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (size_t i = 0; i < size; ++i) {
                        dst_[i] = static_cast<T>(distribution(rd::generator_));
                    }
                }

                template<typename T, typename Td>
                inline void fisher_f(unsigned char * const dst, const Shape &shape, Td numerator_free_degree, Td denominator_free_degree) {
                    std::fisher_f_distribution <Td> distribution(numerator_free_degree, denominator_free_degree);
                    T * const dst_ = reinterpret_cast<T * const >(dst);
                    size_t size = shape.volume();
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (size_t i = 0; i < size; ++i) {
                        dst_[i] = static_cast<T>(distribution(rd::generator_));
                    }
                }
            }
        }
    }
}

#endif /* __U_RANDOM_CPU_HPP__ */
