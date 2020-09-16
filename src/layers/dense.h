#ifndef DENSE_H_
#define DENSE_H_

#include <armadillo>
#include <cassert>
#include <cmath>
#include <vector>

namespace afs {

class Dense {
 public:

  // Construct dense (fully connected layer)
  Dense(size_t num_inputs, size_t num_outputs);

  void Forward(arma::vec& input, arma::vec& output);
  void Backward(arma::vec& upstream_gradient);
  arma::vec GetGradientWrtInput() { return grad_input; }
  void UpdateWeightsAndBiases(size_t batch_size, double learning_rate);

 private:
  size_t num_inputs;
  size_t num_outputs;
  arma::vec input;
  arma::vec output;

  arma::mat weights;
  arma::vec biases;

  arma::vec grad_input;
  arma::mat grad_weights;
  arma::vec grad_biases;

  arma::vec accumulated_grad_input;
  arma::mat accumulated_grad_weights;
  arma::vec accumulated_grad_biases;

  void ResetGradient();
};

}  // namespace afs

#endif