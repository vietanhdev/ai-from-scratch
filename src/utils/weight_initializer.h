#ifndef WEIGHT_INITIALIZER_H_
#define WEIGHT_INITIALIZER_H_

#include <iostream>
#include <string>

#include "random_generator.h"

namespace afs {

class WeightInitializer {

  std::string initializer_name;
  int num_inputs;

 public:
  WeightInitializer(const std::string &initializer_name, int num_inputs) {
    this->initializer_name = initializer_name;
    this->num_inputs = num_inputs;
  }

  double GetRandomWeight() {
    if (initializer_name == "xavier") {
      return RandomGenerator::GetStdNormRandom() * sqrt(1.0 / num_inputs);
    } else if (initializer_name == "he") {
      return RandomGenerator::GetStdNormRandom() * sqrt(2.0 / num_inputs);
    } else if (initializer_name == "small_rand") {
      return RandomGenerator::GetStdNormRandom() * 0.001;
    } else {
      std::cerr << "Wrong weight initializer: " << initializer_name << std::endl;
      exit(1);
    }
    return 0.0;
  }

};

};  // namespace afs

#endif
