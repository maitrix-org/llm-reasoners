/**
 * \file random.h
 *
 * This file defines the basic pseudorandom number generation functionality
 * used by the core library, as well as functions to sample from a number of
 * built-in distributions such as categorical, discrete uniform, continuous
 * uniform, Bernoulli, geometric, beta, gamma, and Dirichlet.
 *
 * <!-- Created on: Aug 15, 2016
 *          Author: asaparov -->
 */

#ifndef RANDOM_H_
#define RANDOM_H_

#include <random>
#include <sstream>

#include "timer.h"
#include "array.h"
#include "io.h"

namespace core {


#if defined(NDEBUG)
static unsigned int seed = (unsigned int) milliseconds();
#else
static unsigned int seed = 0;
#endif

static thread_local std::minstd_rand engine = std::minstd_rand(seed);

/**
 * Returns the initial random seed used by all core functions that require pseudorandom number generation.
 */
inline unsigned int get_seed() {
	return seed;
}

/**
 * Sets the seed of the underlying pseudorandom number generator.
 */
inline void set_seed(unsigned int new_seed) {
	engine.seed(new_seed);
	seed = new_seed;
}

/**
 * Reads the state of the pseudorandom number generator from `in`. This is
 * useful to persist the state of the pseudorandom number generator.
 */
template<typename Stream>
inline bool read_random_state(Stream& in)
{
	size_t length;
	if (!read(length, in)) return false;
	char* state = (char*) alloca(sizeof(char) * length);
	if (state == NULL || !read(state, in, (unsigned int) length))
		return false;

	std::stringstream buffer(std::string(state, length));
	buffer >> engine;
	return true;
}

/**
 * Writes the state of the pseudorandom number generator to `out`. This is
 * useful to persist the state of the pseudorandom number generator.
 */
template<typename Stream>
inline bool write_random_state(Stream& out) {
	std::stringstream buffer;
	buffer << engine;
	std::string data = buffer.str();
	return write(data.length(), out) && write(data.c_str(), out, (unsigned int) data.length());
}

/**
 * Samples from a categorical distribution, where the unnormalized probability
 * of returning the index `i` is given by `probability[i]`. This function
 * normalizes and overwrites `probability` with its cumulative distribution
 * function.
 */
template<typename V,
	typename std::enable_if<std::is_floating_point<V>::value>::type* = nullptr>
unsigned int sample_categorical(V* probability, unsigned int length)
{
#if !defined(NDEBUG)
	if (length == 0)
		fprintf(stderr, "sample_categorical WARNING: Specified length is zero.\n");
#endif

	for (unsigned int j = 1; j < length; j++)
		probability[j] += probability[j - 1];

	/* select the new table assignment */
	V random = probability[length - 1] * ((V) engine() / engine.max());
	unsigned int selected_table = length - 1;
	for (unsigned int j = 0; j < length; j++) {
		if (random < probability[j]) {
			selected_table = j;
			break;
		}
	}
	return selected_table;
}

/**
 * Returns the smallest index `i` such that `random < sum from j=0 to i-1 of probability[j]`.
 * Thus, this function implements the inverse cumulative distribution function
 * for the categorical distribution with unnormalized probabilities given by
 * `probability`.
 */
template<typename U, typename V>
inline unsigned int select_categorical(
		const U* probability, V random, unsigned int length)
{
#if !defined(NDEBUG)
	if (length == 0)
		fprintf(stderr, "select_categorical WARNING: Specified length is zero.\n");
#endif

	V aggregator = (V) 0;
	unsigned int selected_table = length - 1;
	for (unsigned int j = 0; j < length; j++) {
		aggregator += probability[j];
		if (random <= aggregator) {
			selected_table = j;
			break;
		}
	}
	return selected_table;
}

/**
 * Samples from a categorical distribution, where the unnormalized probability
 * of returning the index `i` is given by `probability[i]` and its sum is given
 * by the floating-point `sum`. This function doesn't modify `probability`.
 */
template<typename V,
	typename std::enable_if<std::is_floating_point<V>::value>::type* = nullptr>
unsigned int sample_categorical(const V* probability, V sum, unsigned int length)
{
#if !defined(NDEBUG)
	if (length == 0)
		fprintf(stderr, "sample_categorical WARNING: Specified length is zero.\n");
#endif

	/* select the new table assignment */
	V random = sum * ((V) engine() / engine.max());
	return select_categorical(probability, random, length);
}

/**
 * Samples from a categorical distribution, where the unnormalized probability
 * of returning the index `i` is given by `probability[i]` and its sum is given
 * by the unsigned integer `sum`. This function doesn't modify `probability`.
 */
unsigned int sample_categorical(
	const unsigned int* probability,
	unsigned int sum, unsigned int length)
{
#if !defined(NDEBUG)
	if (length == 0)
		fprintf(stderr, "sample_categorical WARNING: Specified length is zero.\n");
#endif

	/* select the new table assignment */
	unsigned int random = engine() % sum;
	return select_categorical(probability, random, length);
}


/** <!--
 * Sampling functions for uniform, beta and Dirichlet distributions. -->
 */

/* forward declarations */

template<typename T> struct array;

/**
 * Returns a sample from the discrete uniform distribution over `{0, ..., n - 1}`.
 */
inline unsigned int sample_uniform(unsigned int n) {
	return engine() % n;
}

/**
 * Returns a sample from the discrete uniform distribution over `elements`.
 */
template<typename T>
inline const T& sample_uniform(const T* elements, unsigned int length) {
	return elements[engine() % length];
}

/**
 * Returns a sample from the discrete uniform distribution over `elements`.
 */
template<typename T, size_t N>
inline const T& sample_uniform(const T (&elements)[N]) {
	return elements[engine() % N];
}

/**
 * Returns a sample from the discrete uniform distribution over `elements`.
 */
template<typename T>
inline const T& sample_uniform(const array<T>& elements) {
	return sample_uniform(elements.data, (unsigned int) elements.length);
}

/**
 * Returns a sample from the continuous uniform distribution over [0, 1].
 */
template<typename V>
inline V sample_uniform() {
	return (V) engine() / engine.max();
}

/**
 * Returns a sample from the Bernoulli distribution: with probability 0.5,
 * `true` is returned, otherwise `false` is returned.
 */
template<typename V>
inline bool sample_bernoulli(const V& p) {
	return sample_uniform<V>() < p;
}

/**
 * Returns a sample from the geometric distribution with success probability `p`.
 */
template<typename V>
inline unsigned int sample_geometric(const V& p) {
	auto geom = std::geometric_distribution<unsigned int>(p);
	return geom(engine);
}

/**
 * Returns a sample from the negative binomial distribution with number of
 * failures `r` and success probability `p`.
 */
template<typename V>
inline double log_probability_negative_binomial(unsigned int x, const V& r, const V& p) {
	return lgamma(x + r) - lgamma(x + 1) - lgamma(r) + r*log(1 - p) + x*log(p);
}

/**
 * Returns a sample from the beta distribution with shape parameter `alpha` and
 * scale parameter 1. This function assumes `alpha > 0`.
 */
template<typename V>
inline V sample_beta(const V& alpha) {
	static std::gamma_distribution<V> first_gamma = std::gamma_distribution<V>(1.0);
	std::gamma_distribution<V> second_gamma = std::gamma_distribution<V>(alpha);
	V first = first_gamma(engine);
	V second = second_gamma(engine);
	return first / (first + second);
}

/**
 * Returns a sample from the beta distribution with shape parameter `alpha` and
 * scale parameter `beta`. This function assumes `alpha > 0` and `beta > 0`.
 */
template<typename V>
inline V sample_beta(const V& alpha, const V& beta) {
	std::gamma_distribution<V> first_gamma = std::gamma_distribution<V>(alpha, 1.0);
	std::gamma_distribution<V> second_gamma = std::gamma_distribution<V>(beta, 1.0);
	V first = first_gamma(engine);
	V second = second_gamma(engine);
	return first / (first + second);
}

/**
 * Returns a sample from the gamma distribution with shape parameter `alpha`
 * and rate parameter `beta`. This function assumes `alpha > 0` and `beta > 0`.
 */
template<typename V>
inline V sample_gamma(const V& alpha, const V& beta) {
	std::gamma_distribution<V> gamma = std::gamma_distribution<V>(alpha, 1.0 / beta);
	return gamma(engine);
}

/**
 * Returns the log probability of drawing the observation `x` from a gamma
 * distribution with shape parameter `alpha` and rate parameter `beta`. This
 * function assumes `x > 0`, `alpha > 0`, and `beta > 0`.
 */
template<typename V>
inline V log_probability_gamma(const V& x, const V& alpha, const V& beta) {
	return alpha * log(beta) - lgamma(alpha) + (alpha - 1) * log(x) - beta * x;
}

/**
 * Samples from the Dirichlet distribution with parameter `alpha` and dimension
 * `length`. The sample is written to `dst`. This function assumes `dst` has
 * size at least `length`, `alpha[i] > 0` for all `i = 0, ..., length - 1`.
 */
template<typename V>
inline void sample_dirichlet(V* dst, const V* alpha, unsigned int length) {
	V sum = 0.0;
	for (unsigned int i = 0; i < length; i++) {
		if (alpha[i] == 0.0) {
			dst[i] = 0.0;
		} else {
			std::gamma_distribution<V> gamma_dist(alpha[i], 1.0);
			V value = gamma_dist(engine);
			dst[i] = value;
			sum += value;
		}
	}

	for (unsigned int i = 0; i < length; i++)
		dst[i] /= sum;
}


} /* namespace core */

#endif /* RANDOM_H_ */
