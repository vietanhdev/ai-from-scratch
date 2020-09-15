#include "relu.h"

#include <armadillo>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace afs {

ReLU::ReLU(size_t input_height, size_t input_width, size_t input_depth)
    : input_height(input_height),
      input_width(input_width),
      input_depth(input_depth) {}

void ReLU::Forward(arma::cube& input, arma::cube& output) {
  output = arma::zeros(arma::size(input));
  output = arma::max(input, output);
  this->input = input;
  this->output = output;
}

void ReLU::Backward(arma::cube upstream_gradient) {
  gradient_wrt_input = input;
  gradient_wrt_input.transform([](double val) { return val > 0 ? 1 : 0; });
  gradient_wrt_input = gradient_wrt_input % upstream_gradient;
}

arma::cube ReLU::GetGradientWrtInput() { return gradient_wrt_input; }

}  // namespace afs
