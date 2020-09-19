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
  // ReLU(x) = max(0, x)
  output = arma::zeros(arma::size(input));
  output = arma::max(input, output);
  this->input = input;
  this->output = output;
}

void ReLU::Backward(arma::cube upstream_gradient) {
  // Derivative of ReLU = 0 if x = 0
  //                    = 1 if x > 0
  // dL/d(ReLU): upstream_gradient
  // dL/dx = d(ReLU)/dx * dL/d(ReLU)
  grad_input = input;
  grad_input.transform([](double val) { return val > 0 ? 1 : 0; });
  grad_input = grad_input % upstream_gradient;
}

arma::cube ReLU::GetGradientWrtInput() { return grad_input; }

}  // namespace afs
