#include "dense.h"

#include <armadillo>
#include <cassert>
#include <cmath>
#include <vector>

#include "initializer.h"

namespace afs {

Dense::Dense(size_t input_height, size_t input_width, size_t input_depth,
             size_t num_outputs)
    : input_height(input_height),
      input_width(input_width),
      input_depth(input_depth),
      num_outputs(num_outputs) {
  // Initialize the weights.
  weights = arma::zeros(num_outputs, input_height * input_width * input_depth);
  weights = weights.imbue([&]() { return Initializer::GetStdNormRandom(); });

  // Initialize the biases
  biases = arma::zeros(num_outputs);

  // Reset accumulated gradients.
  ResetGradient();
}

void Dense::Forward(arma::cube& input, arma::vec& output) {
  // Flatten the input first
  arma::vec flattened = arma::vectorise(input);
  output = (weights * flattened) + biases;

  // Save input, output for calculating gradient
  this->input = input;
  this->output = output;
}

void Dense::Backward(arma::vec& upstream_gradient) {
  arma::vec gradInputVec =
      arma::zeros(input_height * input_width * input_depth);
  for (size_t i = 0; i < (input_height * input_width * input_depth); i++) {
    gradInputVec[i] = arma::dot(weights.col(i), upstream_gradient);
  }

  arma::cube tmp((input_height * input_width * input_depth), 1, 1);
  tmp.slice(0).col(0) = gradInputVec;
  grad_input = arma::reshape(tmp, input_height, input_width, input_depth);

  accumulated_grad_input += grad_input;

  grad_weights = arma::zeros(arma::size(weights));
  for (size_t i = 0; i < grad_weights.n_rows; i++)
    grad_weights.row(i) = vectorise(input).t() * upstream_gradient[i];

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
  accumulated_grad_input = arma::zeros(input_height, input_width, input_depth);
  accumulated_grad_weights =
      arma::zeros(num_outputs, input_height * input_width * input_depth);
  accumulated_grad_biases = arma::zeros(num_outputs);
}

}  // namespace afs
