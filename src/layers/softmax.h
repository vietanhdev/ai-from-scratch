#ifndef SOFTMAX_H_
#define SOFTMAX_H_

#include <armadillo>
#include <iostream>

namespace afs {

class Softmax {
 private:
  size_t num_inputs;
  arma::vec input;
  arma::vec output;

  arma::vec grad_wrt_input;

 public:
  Softmax(size_t num_inputs);
  void Forward(const arma::vec& input, arma::vec& output);
  void Backward(const arma::vec& upstream_gradient);
  arma::vec GetGradientWrtInput();
};

}  // namespace afs

#endif