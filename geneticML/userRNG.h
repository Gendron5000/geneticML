#ifndef USERRNG_H 
#define USERRNG_H 

#include <random>

namespace UserRNG {

// static random engine to use with all distributions
static std::default_random_engine rng(
        (int) std::chrono::system_clock::now().time_since_epoch().count());

// 'most' generic fn that all the other functions will call with their own types
// returns a functor that generates a random number with the given distribution
template <typename T, typename Distribution, typename ...Args>
auto GetRngFn(Args&&... args)
{
	return [dist = Distribution(args...)]() mutable { return dist(rng); };
}

// gets random real numbers (doubles) between the passed in range
template <typename Distribution = std::uniform_real_distribution<double>>
auto GetRngFn(double start, double end)
{
	return GetRngFn<double, Distribution>(start, end);
}

// gets random integer numbers (ints) between the passed in range
template <typename Distribution = std::uniform_int_distribution<int>>
auto GetRngFn(int start, int end)
{
	return GetRngFn<int, Distribution>(start, end);
}

// gets random yes/no (true/false) where the probability of true is
// the passed in argument
template <typename Distribution = std::bernoulli_distribution>
auto GetRngFn(double in_trueProb)
{
	return GetRngFn<bool, Distribution>(in_trueProb);
}

// gets a distribution using passed in weights to set the probability 
// that each number will get chosen
template <typename Distribution = std::discrete_distribution<>>
auto GetRngFn(std::vector<int> in_weights)
{
	return GetRngFn<int, Distribution>(in_weights.begin(), in_weights.end());
}

};

#endif
