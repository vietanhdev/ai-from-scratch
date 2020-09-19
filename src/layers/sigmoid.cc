#include "sigmoid.h"

#include <armadillo>
#include <iostream>

namespace afs {

Sigmoid::Sigmoid(size_t num_inputs) : num_inputs(num_inputs) {}

void Sigmoid::Forward(const arma::vec& input, arma::vec& output) {
  // Sigmoid(x) = 1 / 1 + e^(-x)
  output = 1.0 / (1 + arma::exp(-input));

  this->input = input;
  this->output = output;
}

void Sigmoid::Backward(const arma::vec& upstream_gradient) {
  // Derivative of sigmoid = sigmoid * (1 - sigmoid)
  // dL/d(sigmoid): upstream_gradient
  // dL/dx = d(sigmoid)/dx * dL/d(sigmoid)
  grad_wrt_input = this->output % (1.0 - this->output) % upstream_gradient;
}

arma::vec Sigmoid::GetGradientWrtInput() { return grad_wrt_input; }

}  // namespace afs