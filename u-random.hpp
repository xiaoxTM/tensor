/*
* u-random.hpp
*
*  Created on: 2017/01/08
*      Author: xiaox
*/

#ifndef __U_TENSOR_RANDOM_HPP__
#define __U_TENSOR_RANDOM_HPP__

#include "u-tensor.hpp"

#ifdef USE_CUDA
#include "rd/u-rd-gpu.cu"
#else
#include "rd/u-rd-cpu.hpp"
#endif

namespace u {
    namespace tensor {

        namespace op {

            template<typename T>
            class Bernoulli {
            public:
                static void run(unsigned char * const dst, const Shape &shape, double p) {
                    #ifdef USE_CUDA
                    u::tensor::rd::gpu::bernoulli<T>(dst, shape, p);
                    #else
                    u::tensor::rd::cpu::bernoulli<T>(dst, shape, p);
                    #endif
                }
            };

            template<typename T>
            class Binomial {
            public:
                template<typename Td>
                static void _run(unsigned char * const dst, const Shape &shape, Td top, double p) {
                    #ifdef USE_CUDA
                    u::tensor::rd::gpu::binomial<T, Td>(dst, shape, top, p);
                    #else
                    u::tensor::rd::cpu::binomial<T, Td>(dst, shape, top, p);
                    #endif
                }

                static void run(unsigned char * const dst, const Shape &shape, short top, double p) {
                    _run<short>(dst, shape, top, p);
                }

                static void run(unsigned char * const dst, const Shape &shape, unsigned short top, double p) {
                    _run<unsigned short>(dst, shape, top, p);
                }

                static void run(unsigned char * const dst, const Shape &shape, int top, double p) {
                    _run<int>(dst, shape, top, p);
                }

                static void run(unsigned char * const dst, const Shape &shape, unsigned int top, double p) {
                    _run<unsigned int>(dst, shape, top, p);
                }

                static void run(unsigned char * const dst, const Shape &shape, long top, double p) {
                    _run<long>(dst, shape, top, p);
                }

                static void run(unsigned char * const dst, const Shape &shape, unsigned long top, double p) {
                    _run<unsigned long>(dst, shape, top, p);
                }

                static void run(unsigned char * const dst, const Shape &shape, long long top, double p) {
                    _run<long long>(dst, shape, top, p);
                }

                static void run(unsigned char * const dst, const Shape &shape, unsigned long long top, double p) {
                    _run<unsigned long long>(dst, shape, top, p);
                }
            };

            template<typename T>
            class Cauchy {
            public:
                template <typename Td>
                static void _run(unsigned char * const dst, const Shape &shape, Td loc, Td scale) {
                    #ifdef USE_CUDA
                    u::tensor::rd::gpu::cauchy<T, Td>(dst, shape, loc, scale);
                    #else
                    u::tensor::rd::cpu::cauchy<T, Td>(dst, shape, loc, scale);
                    #endif
                }

                static void run(unsigned char * const dst, const Shape &shape, float loc, float scale) {
                    _run<float>(dst, shape, loc, scale);
                }

                static void run(unsigned char * const dst, const Shape &shape, double loc, double scale) {
                    _run<double>(dst, shape, loc, scale);
                }

                static void run(unsigned char * const dst, const Shape &shape, long double loc, long double scale) {
                    _run<long double>(dst, shape, loc, scale);
                }
            };

            template<typename T>
            class ChiSquared {
            public:
                template <typename Td>
                static void _run(unsigned char * const dst, const Shape &shape, Td free_degree) {
                    #ifdef USE_CUDA
                    u::tensor::rd::gpu::chi_squared<T, Td>(dst, shape, free_degree);
                    #else
                    u::tensor::rd::cpu::chi_squared<T, Td>(dst, shape, free_degree);
                    #endif
                }

                static void run(unsigned char * const dst, const Shape &shape, float free_degree) {
                    _run<float>(dst, shape, free_degree);
                }

                static void run(unsigned char * const dst, const Shape &shape, double free_degree) {
                    _run<double>(dst, shape, free_degree);
                }

                static void run(unsigned char * const dst, const Shape &shape, long double free_degree) {
                    _run<long double>(dst, shape, free_degree);
                }
            };

            template<typename T>
            class Discrete {
            public:
                static void run(unsigned char * const dst, const Shape &shape, size_t nw, double xmin, double xmax) {
                    #ifdef USE_CUDA
                    u::tensor::rd::gpu::discrete<T>(dst, shape, nw, xmin, xmax);
                    #else
                    u::tensor::rd::cpu::discrete<T>(dst, shape, nw, xmin, xmax);
                    #endif
                }
            };

            template<typename T>
            class Exponential {
            public:
                template <typename Td>
                static void _run(unsigned char * const dst, const Shape &shape, Td lambda) {
                    #ifdef USE_CUDA
                    u::tensor::rd::gpu::exponential<T, Td>(dst, shape, lambda);
                    #else
                    u::tensor::rd::cpu::exponential<T, Td>(dst, shape, lambda);
                    #endif
                }

                static void run(unsigned char * const dst, const Shape &shape, float lambda) {
                    _run<float>(dst, shape, lambda);
                }

                static void run(unsigned char * const dst, const Shape &shape, double lambda) {
                    _run<double>(dst, shape, lambda);
                }

                static void run(unsigned char * const dst, const Shape &shape, long double lambda) {
                    _run<long double>(dst, shape, lambda);
                }
            };

            template<typename T>
            class Gamma {
            public:
                template <typename Td>
                static void _run(unsigned char * const dst, const Shape &shape, Td alpha, Td beta) {
                    #ifdef USE_CUDA
                    u::tensor::rd::gpu::gamma<T, Td>(dst, shape, alpha, beta);
                    #else
                    u::tensor::rd::cpu::gamma<T, Td>(dst, shape, alpha, beta);
                    #endif
                }

                static void run(unsigned char * const dst, const Shape &shape, float alpha, float beta) {
                    _run<float>(dst, shape, alpha, beta);
                }

                static void run(unsigned char * const dst, const Shape &shape, double alpha, double beta) {
                    _run<double>(dst, shape, alpha, beta);
                }

                static void run(unsigned char * const dst, const Shape &shape, long double alpha, long double beta) {
                    _run<long double>(dst, shape, alpha, beta);
                }
            };

            template<typename T>
            class Geometric {
            public:
                static void run(unsigned char * const dst, const Shape &shape, double p) {
                    #ifdef USE_CUDA
                    u::tensor::rd::gpu::geometric<T>(dst, shape, p);
                    #else
                    u::tensor::rd::cpu::geometric<T>(dst, shape, p);
                    #endif
                }
            };

            template<typename T>
            class LogNormal {
            public:
                template <typename Td>
                static void _run(unsigned char * const dst, const Shape &shape, Td mean, Td stddev) {
                    #ifdef USE_CUDA
                    u::tensor::rd::gpu::lognormal<T, Td>(dst, shape, mean, stddev);
                    #else
                    u::tensor::rd::cpu::lognormal<T, Td>(dst, shape, mean, stddev);
                    #endif
                }

                static void run(unsigned char * const dst, const Shape &shape, float mean, float stddev) {
                    _run<float>(dst, shape, mean, stddev);
                }

                static void run(unsigned char * const dst, const Shape &shape, double mean, double stddev) {
                    _run<double>(dst, shape, mean, stddev);
                }

                static void run(unsigned char * const dst, const Shape &shape, long double mean, long double stddev) {
                    _run<long double>(dst, shape, mean, stddev);
                }
            };

            template<typename T>
            class Normal {
            public:
                template <typename Td>
                static void _run(unsigned char * const dst, const Shape &shape, Td mean, Td stddev) {
                    #ifdef USE_CUDA
                    u::tensor::rd::gpu::normal<T, Td>(dst, shape, mean, stddev);
                    #else
                    u::tensor::rd::cpu::normal<T, Td>(dst, shape, mean, stddev);
                    #endif
                }

                static void run(unsigned char * const dst, const Shape &shape, float mean, float stddev) {
                    _run<float>(dst, shape, mean, stddev);
                }

                static void run(unsigned char * const dst, const Shape &shape, double mean, double stddev) {
                    _run<double>(dst, shape, mean, stddev);
                }

                static void run(unsigned char * const dst, const Shape &shape, long double mean, long double stddev) {
                    _run<long double>(dst, shape, mean, stddev);
                }
            };

            template<typename T>
            class Poisson {
            public:
                template <typename Td>
                static void _run(unsigned char * const dst, const Shape &shape, Td mean) {
                    #ifdef USE_CUDA
                    u::tensor::rd::gpu::poisson<T, Td>(dst, shape, mean);
                    #else
                    u::tensor::rd::cpu::poisson<T, Td>(dst, shape, mean);
                    #endif
                }

                static void run(unsigned char * const dst, const Shape &shape, short mean) {
                    _run<short>(dst, shape, mean);
                }

                static void run(unsigned char * const dst, const Shape &shape, unsigned short mean) {
                    _run<unsigned short>(dst, shape, mean);
                }

                static void run(unsigned char * const dst, const Shape &shape, int mean) {
                    _run<int>(dst, shape, mean);
                }

                static void run(unsigned char * const dst, const Shape &shape, unsigned int mean) {
                    _run<unsigned int>(dst, shape, mean);
                }

                static void run(unsigned char * const dst, const Shape &shape, long mean) {
                    _run<long>(dst, shape, mean);
                }

                static void run(unsigned char * const dst, const Shape &shape, unsigned long mean) {
                    _run<unsigned long>(dst, shape, mean);
                }

                static void run(unsigned char * const dst, const Shape &shape, long long mean) {
                    _run<long long>(dst, shape, mean);
                }

                static void run(unsigned char * const dst, const Shape &shape, unsigned long long mean) {
                    _run<unsigned long long>(dst, shape, mean);
                }
            };

            template<typename T>
            class Student_T {
            public:
                template <typename Td>
                static void _run(unsigned char * const dst, const Shape &shape, Td free_degree) {
                    #ifdef USE_CUDA
                    u::tensor::rd::gpu::student_t<T, Td>(dst, shape, free_degree);
                    #else
                    u::tensor::rd::cpu::student_t<T, Td>(dst, shape, free_degree);
                    #endif
                }

                static void run(unsigned char * const dst, const Shape &shape, float free_degree) {
                    _run<float>(dst, shape, free_degree);
                }

                static void run(unsigned char * const dst, const Shape &shape, double free_degree) {
                    _run<double>(dst, shape, free_degree);
                }

                static void run(unsigned char * const dst, const Shape &shape, long double free_degree) {
                    _run<long double>(dst, shape, free_degree);
                }
            };

            template<typename T>
            class Weibull {
            public:
                template <typename Td>
                static void _run(unsigned char * const dst, const Shape &shape, Td a, Td b) {
                    #ifdef USE_CUDA
                    u::tensor::rd::gpu::weibull<T, Td>(dst, shape, a, b);
                    #else
                    u::tensor::rd::cpu::weibull<T, Td>(dst, shape, a, b);
                    #endif
                }

                static void run(unsigned char * const dst, const Shape &shape, float a, float b) {
                    _run<float>(dst, shape, a, b);
                }

                static void run(unsigned char * const dst, const Shape &shape, double a, double b) {
                    _run<double>(dst, shape, a, b);
                }

                static void run(unsigned char * const dst, const Shape &shape, long double a, long double b) {
                    _run<long double>(dst, shape, a, b);
                }
            };

            template<typename T>
            class UniformInt {
            public:
                template <typename Td>
                static void _run(unsigned char * const dst, const Shape &shape, Td lower, Td upper) {
                    #ifdef USE_CUDA
                    u::tensor::rd::gpu::uniform_int<T, Td>(dst, shape, lower, upper);
                    #else
                    u::tensor::rd::cpu::uniform_int<T, Td>(dst, shape, lower, upper);
                    #endif
                }

                static void run(unsigned char * const dst, const Shape &shape, short lower, short upper) {
                    _run<short>(dst, shape, lower, upper);
                }

                static void run(unsigned char * const dst, const Shape &shape, unsigned short lower, unsigned short upper) {
                    _run<unsigned short>(dst, shape, lower, upper);
                }

                static void run(unsigned char * const dst, const Shape &shape, int lower, int upper) {
                    _run<int>(dst, shape, lower, upper);
                }

                static void run(unsigned char * const dst, const Shape &shape, unsigned int lower, unsigned int upper) {
                    _run<unsigned int>(dst, shape, lower, upper);
                }

                static void run(unsigned char * const dst, const Shape &shape, long lower, long upper) {
                    _run<long>(dst, shape, lower, upper);
                }

                static void run(unsigned char * const dst, const Shape &shape, unsigned long lower, unsigned long upper) {
                    _run<unsigned long>(dst, shape, lower, upper);
                }

                static void run(unsigned char * const dst, const Shape &shape, long long lower, long long upper) {
                    _run<long long>(dst, shape, lower, upper);
                }

                static void run(unsigned char * const dst, const Shape &shape, unsigned long long lower, unsigned long long  upper) {
                    _run<unsigned long long>(dst, shape, lower, upper);
                }
            };

            template<typename T>
            class UniformReal {
            public:
                template <typename Td>
                static void _run(unsigned char * const dst, const Shape &shape, Td lower, Td upper) {
                    #ifdef USE_CUDA
                    u::tensor::rd::gpu::uniform_real<T, Td>(dst, shape, lower, upper);
                    #else
                    u::tensor::rd::cpu::uniform_real<T, Td>(dst, shape, lower, upper);
                    #endif
                }

                static void run(unsigned char * const dst, const Shape &shape, float lower, float upper) {
                    _run<float>(dst, shape, lower, upper);
                }

                static void run(unsigned char * const dst, const Shape &shape, double lower, double upper) {
                    _run<double>(dst, shape, lower, upper);
                }

                static void run(unsigned char * const dst, const Shape &shape, long double lower, long double upper) {
                    _run<long double>(dst, shape, lower, upper);
                }
            };

            template<typename T>
            class Fisher_F {
            public:
                template <typename Td>
                static void _run(unsigned char * const dst, const Shape &shape, Td numerator_free_degree, Td denominator_free_degree) {
                    #ifdef USE_CUDA
                    u::tensor::rd::gpu::fisher_f<T, Td>(dst, shape, numerator_free_degree, denominator_free_degree);
                    #else
                    u::tensor::rd::cpu::fisher_f<T, Td>(dst, shape, numerator_free_degree, denominator_free_degree);
                    #endif
                }

                static void run(unsigned char * const dst, const Shape &shape, float numerator_free_degree, float denominator_free_degree) {
                    _run<float>(dst, shape, numerator_free_degree, denominator_free_degree);
                }

                static void run(unsigned char * const dst, const Shape &shape, double numerator_free_degree, double denominator_free_degree) {
                    _run<double>(dst, shape, numerator_free_degree, denominator_free_degree);
                }

                static void run(unsigned char * const dst, const Shape &shape, long double numerator_free_degree, long double denominator_free_degree) {
                    _run<long double>(dst, shape, numerator_free_degree, denominator_free_degree);
                }
            };
        }

        namespace random {

            static void bernoulli(u::tensor::Tensor &t, double p) {
                op::run<u::tensor::Tensor, op::Bernoulli>(t, p);
            }

            static u::tensor::Tensor bernoulli(const u::tensor::Shape &shape, u::tensor::DType type, double p) {
                u::tensor::Tensor t(shape, type);
                bernoulli(t, p);
                return (t);
            }

            template <typename T>
            static void binomial(u::tensor::Tensor &t, T top, double p) {
                op::run<u::tensor::Tensor, op::Binomial>(t, top, p);
            }

            template <typename T>
            static u::tensor::Tensor binomial(const Shape &shape, u::tensor::DType type, T top, double p) {
                u::tensor::Tensor t(shape, type, false);
                binomial<T>(t, top, p);
                return (t);
            }

            template <typename T>
            static void cauchy(u::tensor::Tensor &t, T loc, T scale) {
                op::run<u::tensor::Tensor, op::Cauchy>(t, loc, scale);
            }

            template <typename T>
            static u::tensor::Tensor cauchy(const Shape &shape, u::tensor::DType type, T loc, T scale) {
                u::tensor::Tensor t(shape, type, false);
                cauchy<T>(t, loc, scale);
                return (t);
            }

            template <typename T>
            static void chi_squared(u::tensor::Tensor &t, T free_degree) {
                op::run<u::tensor::Tensor, op::ChiSquared>(t, free_degree);
            }

            template <typename T>
            static u::tensor::Tensor chi_squared(const Shape &shape, u::tensor::DType type, T free_degree) {
                u::tensor::Tensor ret(shape, type, false);
                chi_squared<T>(ret, free_degree);
                return (ret);
            }

            template <typename T>
            static void discrete(u::tensor::Tensor &t, size_t nw, double xmin, double xmax) {
                op::run<u::tensor::Tensor, op::Discrete>(t, nw, xmin, xmax);
            }

            template <typename T>
            static u::tensor::Tensor discrete(const Shape &shape, u::tensor::DType type, size_t nw, double xmin, double xmax) {
                u::tensor::Tensor t(shape, type, false);
                discrete<T>(t, nw, xmin, xmax);
                return (t);
            }

            template <typename T>
            static void exponential(u::tensor::Tensor &t, T lambda) {
                op::run<u::tensor::Tensor, op::Exponential>(t, lambda);
            }

            template <typename T>
            static u::tensor::Tensor exponential(const Shape &shape, u::tensor::DType type, T lambda) {
                u::tensor::Tensor t(shape, type, false);
                exponential<T>(t, lambda);
                return (t);
            }

            template <typename T>
            static void gamma(u::tensor::Tensor &t, T alpha, T belta) {
                op::run<u::tensor::Tensor, op::Gamma>(t, alpha, belta);
            }

            template <typename T>
            static u::tensor::Tensor gamma(const Shape &shape, u::tensor::DType type, T alpha, T belta) {
                u::tensor::Tensor t(shape, type, false);
                gamma<T>(t, alpha, belta);
                return (t);
            }

            template <typename T>
            static void geometric(u::tensor::Tensor &t, double p) {
                op::run<u::tensor::Tensor, op::Geometric>(t, p);
            }

            template <typename T>
            static u::tensor::Tensor geometric(const Shape &shape, u::tensor::DType type, double p) {
                u::tensor::Tensor t(shape, type, false);
                geometric<T>(t, p);
                return (t);
            }

            template <typename T>
            static void lognormal(u::tensor::Tensor &t, T mean, T stddev) {
                op::run<u::tensor::Tensor, op::LogNormal>(t, mean, stddev);
            }

            template <typename T>
            static u::tensor::Tensor lognormal(const Shape &shape, u::tensor::DType type, T mean, T stddev) {
                u::tensor::Tensor t(shape, type, false);
                lognormal<T>(t, mean, stddev);
                return (t);
            }

            template <typename T>
            static void normal(u::tensor::Tensor &t, T mean, T stddev) {
                op::run<u::tensor::Tensor, op::Normal>(t, mean, stddev);
            }

            template <typename T>
            static u::tensor::Tensor normal(const Shape &shape, u::tensor::DType type, T mean, T stddev) {
                u::tensor::Tensor t(shape, type, false);
                normal<T>(t, mean, stddev);
                return (t);
            }

            template <typename T>
            static void poisson(u::tensor::Tensor &t, T mean) {
                op::run<u::tensor::Tensor, op::Poisson>(t, mean);
            }

            template <typename T>
            static u::tensor::Tensor poisson(const Shape &shape, u::tensor::DType type, T mean) {
                u::tensor::Tensor t(shape, type, false);
                poisson<T>(t, mean);
                return (t);
            }

            template <typename T>
            static void student_t(u::tensor::Tensor &t, T free_degree) {
                op::run<u::tensor::Tensor, op::Student_T>(t, free_degree);
            }

            template <typename T>
            static u::tensor::Tensor student_t(const Shape &shape, u::tensor::DType type, T free_degree) {
                u::tensor::Tensor t(shape, type, false);
                student_t<T>(t, free_degree);
                return (t);
            }

            template <typename T>
            static void weibull(u::tensor::Tensor &t, T a, T b) {
                op::run<u::tensor::Tensor, op::Weibull>(t, a, b);
            }

            template <typename T>
            static u::tensor::Tensor weibull(const Shape &shape, u::tensor::DType type, T a, T b) {
                u::tensor::Tensor t(shape, type, false);
                weibull<T>(t, a, b);
                return (t);
            }

            template <typename T>
            static void uniform_int(u::tensor::Tensor &t, T lower, T upper) {
                op::run<u::tensor::Tensor, op::UniformInt>(t, lower, upper);
            }

            template <typename T>
            static u::tensor::Tensor uniform_int(const Shape &shape, u::tensor::DType type, T lower, T upper) {
                u::tensor::Tensor t(shape, type, false);
                uniform_int<T>(t, lower, upper);
                return (t);
            }

            template <typename T>
            static void uniform_real(u::tensor::Tensor &t, T lower, T upper) {
                op::run<u::tensor::Tensor, op::UniformReal>(t, lower, upper);
            }

            template <typename T>
            static u::tensor::Tensor uniform_real(const Shape &shape, u::tensor::DType type, T lower, T upper) {
                u::tensor::Tensor t(shape, type, false);
                uniform_real<T>(t, lower, upper);
                return (t);
            }

            template <typename T>
            static void fisher_f(u::tensor::Tensor &t, T numerator_free_degree, T denominator_free_degree) {
                op::run<u::tensor::Tensor, op::Fisher_F>(t, numerator_free_degree, denominator_free_degree);
            }

            template <typename T>
            static u::tensor::Tensor fisher_f(const Shape &shape, u::tensor::DType type, T numerator_free_degree, T denominator_free_degree) {
                u::tensor::Tensor t(shape, type, false);
                uniform_real<T>(t, numerator_free_degree, denominator_free_degree);
                return (t);
            }
        }
    }
}
#endif /* __U_RANDOM_HPP__ */
