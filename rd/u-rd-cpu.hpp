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

static std::default_random_engine generator_;

}

namespace cpu {

template<typename T>
inline void random_bernoulli(unsigned char * const dst, size_t size,
		double p) {
	//size_t code = typeid(T).hash_code();
	std::bernoulli_distribution distribution(p);
	T * const dst_ = reinterpret_cast<T * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = static_cast<T>(distribution(rd::generator_));
	}
}

template<typename Td, typename Tr>
inline void random_binomial(unsigned char * const dst, size_t size,
		Td top, double p) {
	std::binomial_distribution < Td > distribution(top, p);
	Tr * const dst_ = reinterpret_cast<Tr * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = static_cast<Tr>(distribution(rd::generator_));
	}
}

template<typename Td, typename Tr>
inline void random_cauchy(unsigned char * const dst, size_t size, Td loc,
		Td scale) {
	std::cauchy_distribution < Td > distribution(loc, scale);
	Tr * const dst_ = reinterpret_cast<Tr * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = static_cast<Tr>(distribution(rd::generator_));
	}
}

template<typename Td, typename Tr>
inline void random_chi_squared(unsigned char * const dst, size_t size,
		Td free_degree) {
	std::chi_squared_distribution < Td > distribution(free_degree);
	Tr * const dst_ = reinterpret_cast<Tr * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = static_cast<Tr>(distribution(rd::generator_));
	}
}

template<typename Td, typename Tr>
inline void random_discrete(unsigned char * const dst, size_t size,
		size_t nw, double xmin, double xmax) {
	std::discrete_distribution<Td> distribution(nw, xmin, xmax);
	Tr * const dst_ = reinterpret_cast<Tr * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = static_cast<Tr>(distribution(rd::generator_));
	}
}

template<typename Td, typename Tr>
inline void random_exponential(unsigned char * const dst, size_t size,
		Td lambda) {
	std::exponential_distribution < Td > distribution(lambda);
	Tr * const dst_ = reinterpret_cast<Tr * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = static_cast<Tr>(distribution(rd::generator_));
	}
}

template<typename Td, typename Tr>
inline void random_gamma(unsigned char * const dst, size_t size,
		Td alpha, Td belta) {
	std::gamma_distribution < Td > distribution(alpha, belta);
	Tr * const dst_ = reinterpret_cast<Tr * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = static_cast<Tr>(distribution(rd::generator_));
	}
}

template<typename Td, typename Tr>
inline void random_geometric(unsigned char * const dst, size_t size,
		double p) {
	std::geometric_distribution<Td> distribution(p);
	Tr * const dst_ = reinterpret_cast<Tr * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = static_cast<Tr>(distribution(rd::generator_));
	}
}

template<typename Td, typename Tr>
inline void random_lognormal(unsigned char * const dst, size_t size,
		Td mean, Td stddev) {
	std::lognormal_distribution < Td > distribution(mean, stddev);
	Tr * const dst_ = reinterpret_cast<Tr * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = static_cast<Tr>(distribution(rd::generator_));
	}
}

template<typename Td, typename Tr>
inline void random_normal(unsigned char * const dst, size_t size,
		Td mean, Td stddev) {
	std::normal_distribution < Td > distribution(mean, stddev);
	Tr * const dst_ = reinterpret_cast<Tr * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = static_cast<Tr>(distribution(rd::generator_));
	}
}

template<typename Td, typename Tr>
inline void random_poisson(unsigned char * const dst, size_t size,
		Td mean) {
	std::poisson_distribution < Td > distribution(mean);
	Tr * const dst_ = reinterpret_cast<Tr * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = static_cast<Tr>(distribution(rd::generator_));
	}
}

template<typename Td, typename Tr>
inline void random_student_t(unsigned char * const dst, size_t size,
		Td free_degree) {
	std::student_t_distribution < Td > distribution(free_degree);
	Tr * const dst_ = reinterpret_cast<Tr * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = static_cast<Tr>(distribution(rd::generator_));
	}
}

template<typename Td, typename Tr>
inline void random_weibull(unsigned char * const dst, size_t size, Td a,
		Td b) {
	std::weibull_distribution < Td > distribution(a, b);
	Tr * const dst_ = reinterpret_cast<Tr * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = static_cast<Tr>(distribution(rd::generator_));
	}
}

template<typename Td, typename Tr>
inline void random_uniform_int(unsigned char * const dst, size_t size,
		Td lower, Td upper) {
	std::uniform_int_distribution < Td > distribution(lower, upper);
	Tr * const dst_ = reinterpret_cast<Tr * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = static_cast<Tr>(distribution(rd::generator_));
	}
}

template<typename Td, typename Tr>
inline void random_uniform_real(unsigned char * const dst, size_t size,
		Td lower, Td upper) {
	std::uniform_real_distribution < Td > distribution(lower, upper);
	Tr * const dst_ = reinterpret_cast<Tr * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = static_cast<Tr>(distribution(rd::generator_));
	}
}

template<typename Td, typename Tr>
inline void random_fisher_f(unsigned char * const dst, size_t size,
		Td numerator_free_degree, Td denominator_free_degree) {
	std::fisher_f_distribution < Td
			> distribution(numerator_free_degree, denominator_free_degree);
	Tr * const dst_ = reinterpret_cast<Tr * const >(dst);
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < size; ++i) {
		dst_[i] = static_cast<Tr>(distribution(rd::generator_));
	}
}

}
}

#endif /* __U_RANDOM_CPU_HPP__ */
