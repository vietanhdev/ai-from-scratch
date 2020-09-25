#include "max_pooling.h"

#include <armadillo>
#include <cassert>
#include <iostream>

namespace afs {

MaxPooling::MaxPooling(size_t input_height, size_t input_width,
                       size_t input_depth, size_t pooling_window_height,
                       size_t pooling_window_width, size_t vertical_stride,
                       size_t horizontal_stride)
    : input_height(input_height),
      input_width(input_width),
      input_depth(input_depth),
      pooling_window_height(pooling_window_height),
      pooling_window_width(pooling_window_width),
      vertical_stride(vertical_stride),
      horizontal_stride(horizontal_stride) {}

void MaxPooling::Forward(arma::cube& input, arma::cube& output) {
  assert((input_height - pooling_window_height) % vertical_stride == 0);
  assert((input_width - pooling_window_width) % horizontal_stride == 0);
  output = arma::zeros(
                        (input_height - pooling_window_height) / vertical_stride + 1,
                        (input_width - pooling_window_width) / horizontal_stride + 1,
                        input_depth
                      );

  #pragma omp parallel for
  for (size_t i = 0; i < input_depth; ++i) {
    #pragma omp parallel for
    for (size_t j = 0; j <= input_height - pooling_window_height; j += vertical_stride) {
      #pragma omp parallel for
      for (size_t k = 0; k <= input_width - pooling_window_width; k += horizontal_stride) {
        output.slice(i)(j / vertical_stride, k / horizontal_stride) =
            input.slice(i)
                .submat(j, k, j + pooling_window_height - 1,
                        k + pooling_window_width - 1)
                .max();
      }
    }
  }

  this->input = input;
  this->output = output;
}

void MaxPooling::Backward(arma::cube& upstream_gradient) {
  assert(upstream_gradient.n_rows == output.n_rows);
  assert(upstream_gradient.n_cols == output.n_cols);
  assert(upstream_gradient.n_slices == output.n_slices);

  grad_input = arma::zeros(input_height, input_width, input_depth);
  #pragma omp parallel for
  for (size_t i = 0; i < input_depth; ++i) {
    for (size_t j = 0; j + pooling_window_height <= input_height; j += vertical_stride) {
      for (size_t k = 0; k + pooling_window_width <= input_width; k += horizontal_stride) {
        arma::mat tmp(pooling_window_height, pooling_window_width, arma::fill::zeros);
        tmp(input.slice(i)
                .submat(j, k,
                  j + pooling_window_height - 1,
                  k + pooling_window_width - 1
                ).index_max()) =
            upstream_gradient.slice(i)(j / vertical_stride, k / horizontal_stride);
        grad_input.slice(i).submat(j, k, j + pooling_window_height - 1,
                                   k + pooling_window_width - 1) += tmp;
      }
    }
  }
}

arma::cube MaxPooling::GetGradientWrtInput() { return grad_input; }

}  // namespace afs