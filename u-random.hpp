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
        namespace rd {
            #ifdef USE_CUDA
            inline void random_bernoulli(unsigned char * const dst, size_t size, double p) {
                gpu::random_bernoulli(dst, size, p);
            }

            template <typename T>
            inline void random_binomial(unsigned char * const dst, size_t size, T top, double p) {
                gpu::random_binomial<T>(dst, size, top, p);
            }

            template <typename T>
            inline void random_cauchy(unsigned char * const dst, size_t size, T loc, T scale) {
                gpu::random_cauchy<T>(dst, size, loc, scale);
            }

            template <typename T>
            inline void random_chi_squared(unsigned char * const dst, size_t size, T free_degree) {
                gpu::random_chi_squared<T>(dst, size, free_degree);
            }

            template <typename T>
            inline void random_discrete(unsigned char * const dst, size_t size, size_t nw, double xmin, double xmax) {
                gpu::random_discrete<T>(dst, size, nw, xmin, xmax);
            }

            template <typename T>
            inline void random_exponential(unsigned char * const dst, size_t size, T lambda) {
                gpu::random_exponential<T>(dst, size, lambda);
            }

            template <typename T>
            inline void random_gamma(unsigned char * const dst, size_t size, T alpha, T belta) {
                gpu::random_gamma<T>(dst, size, alpha, belta);
            }

            template <typename T>
            inline void random_geometric(unsigned char * const dst, size_t size, double p) {
                gpu::random_geometric<T>(dst, size, p);
            }

            template <typename T>
            inline void random_lognormal(unsigned char * const dst, size_t size, T mean, T stddev) {
                gpu::random_lognormal<T>(dst, size, mean, stddev);
            }

            template <typename T>
            inline void random_normal(unsigned char * const dst, size_t size, T mean, T stddev) {
                gpu::random_normal<T>(dst, size, mean, stddev);
            }

            template <typename T>
            inline void random_poisson(unsigned char * const dst, size_t size, T mean) {
                gpu::random_normal<T>(dst, size, mean);
            }

            template <typename T>
            inline void random_student_t(unsigned char * const dst, size_t size, T free_degree) {
                gpu::random_student_t<T>(dst, size, free_degree);
            }

            template <typename T>
            inline void random_weibull(unsigned char * const dst, size_t size, T a, T b) {
                gpu::random_weibull<T>(dst, size, a, b);
            }

            template <typename T>
            inline void random_uniform_int(unsigned char * const dst, size_t size, T lower, T upper) {
                gpu::random_uniform_int<T>(dst, size, lower, upper);
            }

            template <typename T>
            inline void random_uniform_real(unsigned char * const dst, size_t size, T lower, T upper) {
                gpu::random_uniform_real<T>(dst, size, lower, upper);
            }

            template <typename T>
            inline void random_fisher_f(unsigned char * const dst, size_t size, T numerator_free_degree, T denominator_free_degree) {
                gpu::random_fisher_f<T>(dst, size, numerator_free_degree, denominator_free_degree);
            }

            #else

            template<typename T>
            inline void random_bernoulli(unsigned char * const dst, size_t size, double p) {
                cpu::random_bernoulli<T>(dst, size, p);
            }

            template<typename Td, typename Tr>
            inline void random_binomial(unsigned char * const dst, size_t size, Tr top, double p) {
                cpu::random_binomial<Td, Tr>(dst, size, top, p);
            }

            template<typename Td, typename Tr>
            inline void random_cauchy(unsigned char * const dst, size_t size, Td loc, Td scale) {
                cpu::random_cauchy<Td, Tr>(dst, size, loc, scale);
            }

            template<typename Td, typename Tr>
            inline void random_chi_squared(unsigned char * const dst, size_t size, Td free_degree) {
                cpu::random_chi_squared<Td, Tr>(dst, size, free_degree);
            }

            template<typename Td, typename Tr>
            inline void random_discrete(unsigned char * const dst, size_t size, size_t nw, double xmin, double xmax) {
                cpu::random_discrete<Td, Tr>(dst, size, nw, xmin, xmax);
            }

            template<typename Td, typename Tr>
            inline void random_exponential(unsigned char * const dst, size_t size, Td lambda) {
                cpu::random_exponential<Td, Tr>(dst, size, lambda);
            }

            template<typename Td, typename Tr>
            inline void random_gamma(unsigned char * const dst, size_t size, Td alpha, Td belta) {
                cpu::random_gamma<Td, Tr>(dst, size, alpha, belta);
            }

            template<typename Td, typename Tr>
            inline void random_geometric(unsigned char * const dst, size_t size, double p) {
                cpu::random_geometric<Td, Tr>(dst, size, p);
            }

            template<typename Td, typename Tr>
            inline void random_lognormal(unsigned char * const dst, size_t size, Td mean, Td stddev) {
                cpu::random_lognormal<Td, Tr>(dst, size, mean, stddev);
            }

            template<typename Td, typename Tr>
            inline void random_normal(unsigned char * const dst, size_t size, Td mean, Td stddev) {
                cpu::random_normal<Td, Tr>(dst, size, mean, stddev);
            }

            template<typename Td, typename Tr>
            inline void random_poisson(unsigned char * const dst, size_t size, Td mean) {
                cpu::random_poisson<Td, Tr>(dst, size, mean);
            }

            template<typename Td, typename Tr>
            inline void random_student_t(unsigned char * const dst, size_t size, Td free_degree) {
                cpu::random_student_t<Td, Tr>(dst, size, free_degree);
            }

            template<typename Td, typename Tr>
            inline void random_weibull(unsigned char * const dst, size_t size, Td a, Td b) {
                cpu::random_weibull<Td, Tr>(dst, size, a, b);
            }

            template<typename Td, typename Tr>
            inline void random_uniform_int(unsigned char * const dst, size_t size, Td lower, Td upper) {
                cpu::random_uniform_int<Td, Tr>(dst, size, lower, upper);
            }

            template<typename Td, typename Tr>
            inline void random_uniform_real(unsigned char * const dst, size_t size, Td lower, Td upper) {
                cpu::random_uniform_real<Td, Tr>(dst, size, lower, upper);
            }

            template<typename Td, typename Tr>
            inline void random_fisher_f(unsigned char * const dst, size_t size, Td numerator_free_degree, Td denominator_free_degree) {
                cpu::random_fisher_f<Td, Tr>(dst, size, numerator_free_degree, denominator_free_degree);
            }

            #endif
        }

        namespace chi {

            template<typename T>
            class random_bernoulli_helper {
            public:
                static void run(unsigned char * const dst, size_t size, double p) {
                    rd::random_bernoulli<T>(dst, size, p);
                }
            };

            template<typename Td, typename Tr>
            class random_binomial_helper {
            public:
                static void run(unsigned char * const dst, size_t size, Td top, double p) {
                    rd::random_binomial<Td, Tr>(dst, size, top, p);
                }
            };

            template<typename Td, typename Tr>
            class random_cauchy_helper {
            public:
                static void run(unsigned char * const dst, size_t size, Td loc, Td scale) {
                    rd::random_cauchy<Td, Tr>(dst, size, loc, scale);
                }
            };

            template<typename Td, typename Tr>
            class random_chi_squared_helper {
            public:
                static void run(unsigned char * const dst, size_t size, Td free_degree) {
                    rd::random_chi_squared<Td, Tr>(dst, size, free_degree);
                }
            };

            template<typename Td, typename Tr>
            class random_discrete_helper {
            public:
                static void run(unsigned char * const dst, size_t size, size_t nw, double xmin, double xmax) {
                    rd::random_discrete<Td, Tr>(dst, size, nw, xmin, xmax);
                }
            };

            template<typename Td, typename Tr>
            class random_exponential_helper {
            public:
                static void run(unsigned char * const dst, size_t size, Td lambda) {
                    rd::random_exponential<Td, Tr>(dst, size, lambda);
                }
            };

            template<typename Td, typename Tr>
            class random_gamma_helper {
            public:
                static void run(unsigned char * const dst, size_t size, Td alpha, Td belta) {
                    rd::random_gamma<Td, Tr>(dst, size, alpha, belta);
                }
            };

            template<typename Td, typename Tr>
            class random_geometric_helper {
            public:
                static void run(unsigned char * const dst, size_t size, double p) {
                    rd::random_geometric<Td, Tr>(dst, size, p);
                }
            };

            template<typename Td, typename Tr>
            class random_lognormal_helper {
            public:
                static void run(unsigned char * const dst, size_t size, Td mean, Td stddev) {
                    rd::random_lognormal<Td, Tr>(dst, size, mean, stddev);
                }
            };

            template<typename Td, typename Tr>
            class random_normal_helper {
            public:
                static void run(unsigned char * const dst, size_t size, Td mean, Td stddev) {
                    rd::random_normal<Td, Tr>(dst, size, mean, stddev);
                }
            };

            template<typename Td, typename Tr>
            class random_poisson_helper {
            public:
                static void run(unsigned char * const dst, size_t size, Td mean) {
                    rd::random_poisson<Td, Tr>(dst, size, mean);
                }
            };

            template<typename Td, typename Tr>
            class random_student_t_helper {
            public:
                static void run(unsigned char * const dst, size_t size, Td free_degree) {
                    rd::random_student_t<Td, Tr>(dst, size, free_degree);
                }
            };

            template<typename Td, typename Tr>
            class random_weibull_helper {
            public:
                static void run(unsigned char * const dst, size_t size, Td a, Td b) {
                    rd::random_weibull<Td, Tr>(dst, size, a, b);
                }
            };

            template<typename Td, typename Tr>
            class random_uniform_int_helper {
            public:
                static void run(unsigned char * const dst, size_t size, Td lower, Td upper) {
                    rd::random_uniform_int<Td, Tr>(dst, size, lower, upper);
                }
            };

            template<typename Td, typename Tr>
            class random_uniform_real_helper {
            public:
                static void run(unsigned char * const dst, size_t size, Td lower, Td upper) {
                    rd::random_uniform_real<Td, Tr>(dst, size, lower, upper);
                }
            };

            template<typename Td, typename Tr>
            class random_fisher_f_helper {
            public:
                static void run(unsigned char * const dst, size_t size, Td numerator_free_degree, Td denominator_free_degree) {
                    rd::random_fisher_f<Td, Tr>(dst, size, numerator_free_degree, denominator_free_degree);
                }
            };
        }

        namespace random {

            static void bernoulli(tensor &t, double p) {
                dtype_atomic_run<chi::random_bernoulli_helper>(t.type(), t.ref(), t.size(), p);
            }

            static tensor bernoulli(const std::vector<unsigned int> &shape, dtype_atomic type, double p) {
                tensor ret(shape, type, false);
                bernoulli(ret, p);
                return (ret);
            }

            template <typename T>
            static void binomial(tensor &t, T top, double p) {
                dtype_atomic_run2_<T, chi::random_binomial_helper>(t.type(), t.ref(), t.size(), top, p);
            }

            template <typename T>
            static tensor binomial(const std::vector<unsigned int> &shape, dtype_atomic type, T top, double p) {
                tensor ret(shape, type, false);
                binomial<T>(ret, top, p);
                return (ret);
            }

            template <typename T>
            static void cauchy(tensor &t, T loc, T scale) {
                dtype_atomic_run2_<T, chi::random_cauchy_helper>(t.type(), t.ref(), t.size(), loc, scale);
            }

            template <typename T>
            static tensor cauchy(const std::vector<unsigned int> &shape, dtype_atomic type, T loc, T scale) {
                tensor ret(shape, type, false);
                cauchy<T>(ret, loc, scale);
                return (ret);
            }

            template <typename T>
            static void chi_squared(tensor &t, T free_degree) {
                dtype_atomic_run2_<T, chi::random_chi_squared_helper>(t.type(), t.ref(), t.size(), free_degree);
            }

            template <typename T>
            static tensor chi_squared(const std::vector<unsigned int> &shape, dtype_atomic type, T free_degree) {
                tensor ret(shape, type, false);
                chi_squared<T>(ret, free_degree);
                return (ret);
            }

            template <typename T>
            static void discrete(tensor &t, size_t nw, double xmin, double xmax) {
                dtype_atomic_run2_<T, chi::random_discrete_helper>(t.type(), t.ref(), t.size(), nw, xmin, xmax);
            }

            template <typename T>
            static tensor discrete(const std::vector<unsigned int> &shape, dtype_atomic type, size_t nw, double xmin, double xmax) {
                tensor ret(shape, type, false);
                discrete<T>(ret, nw, xmin, xmax);
                return (ret);
            }

            template <typename T>
            static void exponential(tensor &t, T lambda) {
                dtype_atomic_run2_<T, chi::random_exponential_helper>(t.type(), t.ref(), t.size(), lambda);
            }

            template <typename T>
            static tensor exponential(const std::vector<unsigned int> &shape, dtype_atomic type, T lambda) {
                tensor ret(shape, type, false);
                exponential<T>(ret, lambda);
                return (ret);
            }

            template <typename T>
            static void gamma(tensor &t, T alpha, T belta) {
                dtype_atomic_run2_<T, chi::random_gamma_helper>(t.type(), t.ref(), t.size(), alpha, belta);
            }

            template <typename T>
            static tensor gamma(const std::vector<unsigned int> &shape, dtype_atomic type, T alpha, T belta) {
                tensor ret(shape, type, false);
                gamma<T>(ret, alpha, belta);
                return (ret);
            }

            template <typename T>
            static void geometric(tensor &t, double p) {
                dtype_atomic_run2_<T, chi::random_geometric_helper>(t.type(), t.ref(), t.size(), p);
            }

            template <typename T>
            static tensor geometric(const std::vector<unsigned int> &shape, dtype_atomic type, double p) {
                tensor ret(shape, type, false);
                geometric<T>(ret, p);
                return (ret);
            }

            template <typename T>
            static void lognormal(tensor &t, T mean, T stddev) {
                dtype_atomic_run2_<T, chi::random_lognormal_helper>(t.type(), t.ref(), t.size(), mean, stddev);
            }

            template <typename T>
            static tensor lognormal(const std::vector<unsigned int> &shape, dtype_atomic type, T mean, T stddev) {
                tensor ret(shape, type, false);
                lognormal<T>(ret, mean, stddev);
                return (ret);
            }

            template <typename T>
            static void normal(tensor &t, T mean, T stddev) {
                dtype_atomic_run2_<T, chi::random_normal_helper>(t.type(), t.ref(), t.size(), mean, stddev);
            }

            template <typename T>
            static tensor normal(const std::vector<unsigned int> &shape, dtype_atomic type, T mean, T stddev) {
                tensor ret(shape, type, false);
                normal<T>(ret, mean, stddev);
                return (ret);
            }

            template <typename T>
            static void poisson(tensor &t, T mean) {
                dtype_atomic_run2_<T, chi::random_poisson_helper>(t.type(), t.ref(), t.size(), mean);
            }

            template <typename T>
            static tensor poisson(const std::vector<unsigned int> &shape, dtype_atomic type, T mean) {
                tensor ret(shape, type, false);
                poisson<T>(ret, mean);
                return (ret);
            }

            template <typename T>
            static void student_t(tensor &t, T free_degree) {
                dtype_atomic_run2_<T, chi::random_student_t_helper>(t.type(), t.ref(), t.size(), free_degree);
            }

            template <typename T>
            static tensor student_t(const std::vector<unsigned int> &shape, dtype_atomic type, T free_degree) {
                tensor ret(shape, type, false);
                student_t<T>(ret, free_degree);
                return (ret);
            }

            template <typename T>
            static void weibull(tensor &t, T a, T b) {
                dtype_atomic_run2_<T, chi::random_weibull_helper>(t.type(), t.ref(), t.size(), a, b);
            }

            template <typename T>
            static tensor weibull(const std::vector<unsigned int> &shape, dtype_atomic type, T a, T b) {
                tensor ret(shape, type, false);
                weibull<T>(ret, a, b);
                return (ret);
            }

            template <typename T>
            static void uniform_int(tensor &t, T lower, T upper) {
                dtype_atomic_run2_<T, chi::random_uniform_int_helper>(t.type(), t.ref(), t.size(), lower, upper);
            }

            template <typename T>
            static tensor uniform_int(const std::vector<unsigned int> &shape, dtype_atomic type, T lower, T upper) {
                tensor ret(shape, type, false);
                uniform_int<T>(ret, lower, upper);
                return (ret);
            }

            template <typename T>
            static void uniform_real(tensor &t, T lower, T upper) {
                dtype_atomic_run2_<T, chi::random_uniform_real_helper>(t.type(), t.ref(), t.size(), lower, upper);
            }

            template <typename T>
            static tensor uniform_real(const std::vector<unsigned int> &shape, dtype_atomic type, T lower, T upper) {
                tensor ret(shape, type, false);
                uniform_real<T>(ret, lower, upper);
                return (ret);
            }

            template <typename T>
            static void fisher_f(tensor &t, T numerator_free_degree, T denominator_free_degree) {
                dtype_atomic_run2_<T, chi::random_uniform_real_helper>(t.type(), t.ref(), t.size(), numerator_free_degree, denominator_free_degree);
            }

            template <typename T>
            static tensor fisher_f(const std::vector<unsigned int> &shape, dtype_atomic type, T numerator_free_degree, T denominator_free_degree) {
                tensor ret(shape, type, false);
                uniform_real<T>(ret, numerator_free_degree, denominator_free_degree);
                return (ret);
            }
        }
    }
}
#endif /* __U_RANDOM_HPP__ */
