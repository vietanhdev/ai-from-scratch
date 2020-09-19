#ifndef MAX_POOLING_H_
#define MAX_POOLING_H_

#include <armadillo>
#include <cassert>
#include <iostream>

namespace afs
{

  class MaxPooling
  {

  private:
    size_t input_height;
    size_t input_width;
    size_t input_depth;
    size_t pooling_window_height;
    size_t pooling_window_width;
    size_t vertical_stride;
    size_t horizontal_stride;

  public:
    arma::cube input;
    arma::cube output;
    arma::cube grad_input;

  public:
    MaxPooling(size_t input_height, size_t input_width, size_t input_depth,
               size_t pooling_window_height, size_t pooling_window_width,
               size_t vertical_stride, size_t horizontal_stride);

    void Forward(arma::cube &input, arma::cube &output);
    void Backward(arma::cube &upstream_gradient);

    arma::cube GetGradientWrtInput();

  };

} // namespace afs

#endif