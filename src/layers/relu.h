#ifndef RELU_H_
#define RELU_H_

#include <armadillo>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace afs {

class ReLU {
 private:
  size_t input_height;
  size_t input_width;
  size_t input_depth;

  arma::cube input;
  arma::cube output;

  arma::cube grad_input;

 public:
  ReLU(size_t input_height, size_t input_width, size_t input_depth);
  void Forward(arma::cube& input, arma::cube& output);
  void Backward(arma::cube upstream_gradient);

  arma::cube GetGradientWrtInput();
};

}  // namespace afs

#endif  // RELU_H_
