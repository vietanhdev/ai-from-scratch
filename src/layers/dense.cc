#include "dense.h"

#include <armadillo>
#include <cassert>
#include <cmath>
#include <vector>

#include "initializer.h"

namespace afs {

Dense::Dense(size_t num_inputs, size_t num_outputs)
    : num_inputs(num_inputs), num_outputs(num_outputs) {
  // Initialize the weights.
  weights = arma::zeros(num_outputs, num_inputs);
  weights = weights.imbue([&]() { return Initializer::GetStdNormRandom(); });

  // Initialize the biases
  biases = arma::zeros(num_outputs);

  // Reset accumulated gradients.
  ResetGradient();
}

void Dense::Forward(arma::vec& input, arma::vec& output) {

  output = (weights * input) + biases;

  // Save input, output for calculating gradient
  this->input = input;
  this->output = output;
}

void Dense::Backward(arma::vec& upstream_gradient) {
  // Calculate input gradient
  arma::vec grad_input_vec = arma::zeros(num_inputs);
  for (size_t i = 0; i < num_inputs; i++) {
    grad_input_vec[i] = arma::dot(weights.col(i), upstream_gradient);
  }

  grad_input = grad_input_vec;
  accumulated_grad_input += grad_input_vec;

  // Calculate weight gradient
  grad_weights = arma::zeros(arma::size(weights));
  for (size_t i = 0; i < grad_weights.n_rows; i++) {
    grad_weights.row(i) = vectorise(input).t() * upstream_gradient[i];
  }

  accumulated_grad_weights += grad_weights;

  grad_biases = upstream_gradient;
  accumulated_grad_biases += grad_biases;
}

void Dense::UpdateWeightsAndBiases(size_t batch_size, double learning_rate) {
  weights = weights - learning_rate * (accumulated_grad_weights / batch_size);
  biases = biases - learning_rate * (accumulated_grad_biases / batch_size);
  ResetGradient();
}

void Dense::ResetGradient() {
  accumulated_grad_input = arma::zeros(num_inputs);
  accumulated_grad_weights = arma::zeros(num_outputs, num_inputs);
  accumulated_grad_biases = arma::zeros(num_outputs);
}

}  // namespace afs
