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
  output =
      arma::zeros((input_height - pooling_window_height) / vertical_stride + 1,
                  (input_width - pooling_window_width) / horizontal_stride + 1,
                  input_depth);
  for (size_t sidx = 0; sidx < input_depth; sidx++) {
    for (size_t ridx = 0; ridx <= input_height - pooling_window_height;
         ridx += vertical_stride) {
      for (size_t cidx = 0; cidx <= input_width - pooling_window_width;
           cidx += horizontal_stride) {
        output.slice(sidx)(ridx / vertical_stride, cidx / horizontal_stride) =
            input.slice(sidx)
                .submat(ridx, cidx, ridx + pooling_window_height - 1,
                        cidx + pooling_window_width - 1)
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

  gradient_wrt_input = arma::zeros(input_height, input_width, input_depth);
  for (size_t i = 0; i < input_depth; i++) {
    for (size_t r = 0; r + pooling_window_height <= input_height;
         r += vertical_stride) {
      for (size_t c = 0; c + pooling_window_width <= input_width;
           c += horizontal_stride) {
        arma::mat tmp(pooling_window_height, pooling_window_width,
                      arma::fill::zeros);
        tmp(input.slice(i)
                .submat(r, c, r + pooling_window_height - 1,
                        c + pooling_window_width - 1)
                .index_max()) =
            upstream_gradient.slice(i)(r / vertical_stride,
                                       c / horizontal_stride);
        gradient_wrt_input.slice(i).submat(r, c, r + pooling_window_height - 1,
                                           c + pooling_window_width - 1) += tmp;
      }
    }
  }
}

arma::cube MaxPooling::GetGradientWrtInput() { return gradient_wrt_input; }

}  // namespace afs