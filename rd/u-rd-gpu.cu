/*
 * u-rd.hpp
 *
 *  Created on: 2017/01/08
 *      Author: xiaox
 */

#ifndef __U_TENSOR_RANDOM_CPU_HPP__
#define __U_TENSOR_RANDOM_CPU_HPP__

#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace u {

namespace rd {

std::default_random_engine generator_;

}

namespace cpu {

template <typename T>
inline void random_bernoulli(unsigned char * const dst, size_t size,
		double p) {
	T * const dst_ = reinterpret_cast<T * const>(dst);
	std::bernoulli_distribution distribution(p);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = static_cast<T>(distribution(rd::generator_) ? 1 : 0);
	}
}

template<typename T>
inline void random_binomial(unsigned char * const dst, size_t size,
		T top, double p) {
	std::binomial_distribution distribution<T>(top, p);
	T * const dst_ = reinterpret_cast<T * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = distribution(rd::generator_);
	}
}

template<typename T>
inline void random_cauchy(unsigned char * const dst, size_t size,
		T loc, T scale) {
	std::cauchy_distribution distribution<T>(loc, scale);
	T * const dst_ = reinterpret_cast<T * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = distribution(rd::generator_);
	}
}

template<typename T>
inline void random_chi_square(unsigned char * const dst, size_t size, T free_degree) {
	std::chi_square_distribution distribution<T>(free_degree);
	T * const dst_ = reinterpret_cast<T * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = distribution(rd::generator_);
	}
}

template<typename T>
inline void random_discrete(unsigned char * const dst, size_t size, size_t nw, double xmin, double xmax) {
	std::discrete_distribution distribution<T>(nw, xmin, xmax);
	T * const dst_ = reinterpret_cast<T * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = distribution(rd::generator_);
	}
}

template<typename T>
inline void random_exponential(unsigned char * const dst, size_t size, T lambda) {
	std::exponential_distribution distribution<T>(lambda);
	T * const dst_ = reinterpret_cast<T * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = distribution(rd::generator_);
	}
}

template<typename T>
inline void random_gamma(unsigned char * const dst, size_t size, T alpha, T belta) {
	std::gamma_distribution distribution<T>(alpha, belta);
	T * const dst_ = reinterpret_cast<T * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = distribution(rd::generator_);
	}
}

template<typename T>
inline void random_geometric(unsigned char * const dst, size_t size, double p) {
	std::geometric_distribution distribution<T>(p);
	T * const dst_ = reinterpret_cast<T * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = distribution(rd::generator_);
	}
}

template<typename T>
inline void random_lognormal(unsigned char * const dst, size_t size, T mean, T stddev) {
	std::lognormal_distribution distribution<T>(mean, stddev);
	T * const dst_ = reinterpret_cast<T * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = distribution(rd::generator_);
	}
}

template<typename T>
inline void random_normal(unsigned char * const dst, size_t size, T mean, T stddev) {
	std::normal_distribution distribution<T>(mean, stddev);
	T * const dst_ = reinterpret_cast<T * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = distribution(rd::generator_);
	}
}

template<typename T>
inline void random_poisson(unsigned char * const dst, size_t size, T mean) {
	std::poisson_distribution distribution<T>(mean);
	T * const dst_ = reinterpret_cast<T * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = distribution(rd::generator_);
	}
}

template<typename T>
inline void random_student_t(unsigned char * const dst, size_t size, T free_degree) {
	std::student_t_distribution distribution<T>(free_degree);
	T * const dst_ = reinterpret_cast<T * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = distribution(rd::generator_);
	}
}

template<typename T>
inline void random_weibull(unsigned char * const dst, size_t size, T a, T b) {
	std::weibull_distribution distribution<T>(a, b);
	T * const dst_ = reinterpret_cast<T * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = distribution(rd::generator_);
	}
}

template<typename T>
inline void random_uniform_int(unsigned char * const dst, size_t size, T lower, T upper) {
	std::uniform_int_distribution distribution<T>(lower, upper);
	T * const dst_ = reinterpret_cast<T * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = distribution(rd::generator_);
	}
}

template<typename T>
inline void random_uniform_real(unsigned char * const dst, size_t size, T lower, T upper) {
	std::uniform_real_distribution distribution<T>(lower, upper);
	T * const dst_ = reinterpret_cast<T * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = distribution(rd::generator_);
	}
}

template<typename T>
inline void random_fisher_f(unsigned char * const dst, size_t size, T numerator_free_degree, T denominator_free_degree) {
	std::fisher_f_distribution distribution<T>(numerator_free_degree, denominator_free_degree);
	T * const dst_ = reinterpret_cast<T * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = distribution(rd::generator_);
	}
}

}
}

#endif /* __U_RANDOM_CPU_HPP__ */
