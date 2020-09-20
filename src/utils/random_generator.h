#ifndef RANDOM_GENERATOR_H_
#define RANDOM_GENERATOR_H_

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>

namespace afs {

class RandomGenerator {

 public:
  static double GetStdNormRandom() {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0, 1};
    return d(gen);
  }

  static double GetUniformRandom(double min_value, double max_value) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<double> unif(min_value, max_value);
    return unif(gen);
  }

};

};  // namespace afs

#endif
