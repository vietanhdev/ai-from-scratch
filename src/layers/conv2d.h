#ifndef CONV2D_H_
#define CONV2D_H_

#include <armadillo>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace afs {

class Conv2D {
 private:
  size_t input_height;
  size_t input_width;
  size_t input_depth;
  size_t filter_height;
  size_t filter_width;
  size_t horizontal_stride;
  size_t vertical_stride;
  size_t num_filters;

  std::vector<arma::cube> filters;

  arma::cube input;
  arma::cube output;
  arma::cube grad_input;
  arma::cube accumulated_grad_input;
  std::vector<arma::cube> grad_filters;
  std::vector<arma::cube> accumulated_grad_filters;

 public:
  Conv2D(size_t input_height, size_t input_width, size_t input_depth,
         size_t filter_height, size_t filter_width, size_t horizontal_stride,
         size_t vertical_stride, size_t num_filters,
         const std::string& weight_initializer = "he");
  void Forward(arma::cube& input, arma::cube& output);
  void Backward(arma::cube& upstream_gradient);
  void UpdateFilterWeights(size_t batch_size, double learning_rate);

  std::vector<arma::cube> GetFilters();
  arma::cube GetGradientWrtInput();
  std::vector<arma::cube> GetGradientWrtFilters();

 private:
  void ResetGradient();
};

}  // namespace afs

#endif
