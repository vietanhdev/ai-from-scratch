#ifndef SIGMOID_H_
#define SIGMOID_H_

#include <armadillo>
#include <iostream>

namespace afs {

class Sigmoid {
 private:
  size_t num_inputs;
  arma::vec input;
  arma::vec output;

  arma::vec grad_wrt_input;

 public:
  Sigmoid(size_t num_inputs);
  void Forward(const arma::vec& input, arma::vec& output);
  void Backward(const arma::vec& upstream_gradient);
  arma::vec GetGradientWrtInput();
};

}  // namespace afs

#endif