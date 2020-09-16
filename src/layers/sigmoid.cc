#include "sigmoid.h"

#include <armadillo>
#include <iostream>

namespace afs {

Sigmoid::Sigmoid(size_t num_inputs) : num_inputs(num_inputs) {}

void Sigmoid::Forward(const arma::vec& input, arma::vec& output) {
  output = 1.0 / (1 + arma::exp(-input));

  this->input = input;
  this->output = output;
}

void Sigmoid::Backward(arma::vec& upstream_gradient) {
  grad_wrt_input = this->output * (1.0 - this->output);
}

arma::vec Sigmoid::GetGradientWrtInput() { return grad_wrt_input; }

}  // namespace afs