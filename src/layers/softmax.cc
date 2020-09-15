#include "softmax.h"

#include <armadillo>
#include <iostream>

namespace afs {

Softmax::Softmax(size_t num_inputs) : num_inputs(num_inputs) {}

void Softmax::Forward(arma::vec& input, arma::vec& output) {
  double sum_exp = arma::accu(arma::exp(input - arma::max(input)));
  output = arma::exp(input - arma::max(input)) / sum_exp;

  this->input = input;
  this->output = output;
}

void Softmax::Backward(arma::vec& upstream_gradient) {
  double sub = arma::dot(upstream_gradient, output);
  grad_wrt_input = (upstream_gradient - sub) % output;
}

arma::vec Softmax::GetGradientWrtInput() { return grad_wrt_input; }

}  // namespace afs