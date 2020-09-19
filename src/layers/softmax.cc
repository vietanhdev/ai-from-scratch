#include "softmax.h"

#include <armadillo>
#include <iostream>

namespace afs {

Softmax::Softmax(size_t num_inputs) : num_inputs(num_inputs) {}

void Softmax::Forward(const arma::vec& input, arma::vec& output) {
  // Softmax function: https://cs231n.github.io/linear-classify/#softmax
  // This version is stable softmax: use `- arma::max(input)`
  // to limit the max value of input - arma::max(input) to 0, thus can prevent
  // overflow
  arma::vec numerator = arma::exp(input - arma::max(input));
  double sum_exp = arma::accu(numerator);
  output = numerator / sum_exp;

  this->input = input;
  this->output = output;
}

void Softmax::Backward(const arma::vec& upstream_gradient) {
  // Simple Softmax: http://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
  // TODO (vietanhdev): Stabled Softmax
  double sub = arma::dot(upstream_gradient, output);
  grad_wrt_input = (upstream_gradient - sub) % output;
}

arma::vec Softmax::GetGradientWrtInput() { return grad_wrt_input; }

}  // namespace afs
