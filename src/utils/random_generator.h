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

};

};  // namespace afs

#endif
