#ifndef DENSE_H_
#define DENSE_H_

#include <armadillo>
#include <cassert>
#include <cmath>
#include <vector>

namespace afs {

class Dense {
 public:
  Dense(size_t input_height, size_t input_width, size_t input_depth,
        size_t num_outputs);

  void Forward(arma::cube& input, arma::vec& output);
  void Backward(arma::vec& upstream_gradient);
  arma::cube GetGradientWrtInput() { return grad_input; }
  void UpdateWeightsAndBiases(size_t batch_size, double learning_rate);

 private:
  size_t input_height;
  size_t input_width;
  size_t input_depth;
  arma::cube input;

  size_t num_outputs;
  arma::vec output;

  arma::mat weights;
  arma::vec biases;

  arma::cube grad_input;
  arma::mat grad_weights;
  arma::vec grad_biases;

  arma::cube accumulated_grad_input;
  arma::mat accumulated_grad_weights;
  arma::vec accumulated_grad_biases;

  void ResetGradient();
};

}  // namespace afs

#endif