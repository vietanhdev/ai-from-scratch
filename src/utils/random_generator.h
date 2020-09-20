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
 private:
  RandomGenerator() {}
  inline static RandomGenerator *instance;
  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<double> std_norm_distribution{0, 1};
  std::uniform_real_distribution<double> std_uniform_distribution{0, 1};

 public:
  static RandomGenerator *GetInstance() {
    if (!RandomGenerator::instance) {
      RandomGenerator::instance = new RandomGenerator;
    }
    return RandomGenerator::instance;
  }

  double GetStdNormRandom() { return std_norm_distribution(gen); }
  double GetStdUniformRandom() { return std_uniform_distribution(gen); }
  double GetUniformRandom(double min_value, double max_value) {
    std::uniform_real_distribution<double> unif(min_value, max_value);
    return unif(gen);
  }
};

};  // namespace afs

#endif
