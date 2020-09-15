#include "conv2d.h"

#include <armadillo>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "initializer.h"

namespace afs {

Conv2D::Conv2D(size_t input_height, size_t input_width, size_t input_depth,
               size_t filter_height, size_t filter_width,
               size_t horizontal_stride, size_t vertical_stride,
               size_t num_filters)
    : input_height(input_height),
      input_width(input_width),
      input_depth(input_depth),
      filter_height(filter_height),
      filter_width(filter_width),
      horizontal_stride(horizontal_stride),
      vertical_stride(vertical_stride),
      num_filters(num_filters) {
  // Initialize the filters.
  filters.resize(num_filters);
  for (size_t i = 0; i < num_filters; i++) {
    filters[i] = arma::zeros(filter_height, filter_width, input_depth);
    filters[i].imbue([&]() {
      // Python: self.W = np.random.randn(
      //       n_filter, self.d_X, h_filter, w_filter) / np.sqrt(n_filter / 2.)
      return Initializer::GetStdNormRandom() / sqrt(num_filters / 2.0);
    });
  }

  ResetGradient();
}

void Conv2D::Forward(arma::cube& input, arma::cube& output) {
  // The filter dimensions and strides must satisfy some contraints for
  // the convolution operation to be well defined.
  assert((input_height - filter_height) % vertical_stride == 0);
  assert((input_width - filter_width) % horizontal_stride == 0);

  // Output initialization.
  output = arma::zeros((input_height - filter_height) / vertical_stride + 1,
                       (input_width - filter_width) / horizontal_stride + 1,
                       num_filters);

  // Perform convolution for each filter.
  for (size_t fidx = 0; fidx < num_filters; fidx++) {
    for (size_t i = 0; i <= input_height - filter_height; i += vertical_stride)
      for (size_t j = 0; j <= input_width - filter_width;
           j += horizontal_stride)
        output((i / vertical_stride), (j / horizontal_stride), fidx) =
            arma::dot(arma::vectorise(
                          input.subcube(i, j, 0, i + filter_height - 1,
                                        j + filter_width - 1, input_depth - 1)),
                      arma::vectorise(filters[fidx]));
  }

  // Store the input and output. This will be needed by the backward pass.
  this->input = input;
  this->output = output;
}

void Conv2D::Backward(arma::cube& upstream_gradient) {
  // Upstream gradient must have same dimensions as the output.
  assert(upstream_gradient.n_slices == num_filters);
  assert(upstream_gradient.n_rows == output.n_rows);
  assert(upstream_gradient.n_cols == output.n_cols);

  // Initialize gradient wrt input. Note that the dimensions are same as those
  // of the input.
  grad_input = arma::zeros(arma::size(input));

  // Compute the gradient wrt input.
  for (size_t sidx = 0; sidx < num_filters; sidx++) {
    for (size_t r = 0; r < output.n_rows; r++) {
      for (size_t c = 0; c < output.n_cols; c++) {
        arma::cube tmp(arma::size(input), arma::fill::zeros);
        tmp.subcube(r * vertical_stride, c * horizontal_stride, 0,
                    (r * vertical_stride) + filter_height - 1,
                    (c * horizontal_stride) + filter_width - 1,
                    input_depth - 1) = filters[sidx];
        grad_input += upstream_gradient.slice(sidx)(r, c) * tmp;
      }
    }
  }

  // Update the accumulated gradient wrt input.
  accumulated_grad_input += grad_input;

  // Initialize the gradient wrt filters.
  grad_filters.clear();
  grad_filters.resize(num_filters);
  for (size_t i = 0; i < num_filters; i++)
    grad_filters[i] = arma::zeros(filter_height, filter_width, input_depth);

  // Compute the gradient wrt filters.
  for (size_t fidx = 0; fidx < num_filters; fidx++) {
    for (size_t r = 0; r < output.n_rows; r++) {
      for (size_t c = 0; c < output.n_cols; c++) {
        arma::cube tmp(arma::size(filters[fidx]), arma::fill::zeros);
        tmp = input.subcube(r * vertical_stride, c * horizontal_stride, 0,
                            (r * vertical_stride) + filter_height - 1,
                            (c * horizontal_stride) + filter_width - 1,
                            input_depth - 1);
        grad_filters[fidx] += upstream_gradient.slice(fidx)(r, c) * tmp;
      }
    }
  }

  // Update the accumulated gradient wrt filters.
  for (size_t fidx = 0; fidx < num_filters; fidx++)
    accumulated_grad_filters[fidx] += grad_filters[fidx];
}

void Conv2D::UpdateFilterWeights(size_t batch_size, double learning_rate) {
  for (size_t fidx = 0; fidx < num_filters; fidx++)
    filters[fidx] -=  learning_rate * (accumulated_grad_filters[fidx] / batch_size);

  ResetGradient();
}

void Conv2D::ResetGradient() {
  accumulated_grad_filters.clear();
  accumulated_grad_filters.resize(num_filters);
  for (size_t fidx = 0; fidx < num_filters; fidx++)
    accumulated_grad_filters[fidx] =
        arma::zeros(filter_height, filter_width, input_depth);
  accumulated_grad_input = arma::zeros(input_height, input_width, input_depth);
}

std::vector<arma::cube> Conv2D::GetFilters() { return this->filters; }
arma::cube Conv2D::GetGradientWrtInput() { return grad_input; }
std::vector<arma::cube> Conv2D::GetGradientWrtFilters() { return grad_filters; }

}  // namespace afs
