#include "dropout.h"

#include "utils/random_generator.h"
#include "utils/data_transformer.h"

namespace afs {

Dropout::Dropout(float keep_prop):keep_prop(keep_prop) {
  assert(keep_prop > 0 && keep_prop <= 1);
}

void Dropout::Forward(const arma::cube& input, arma::cube& output,
                      const DropoutMode mode) {
  if (mode == DropoutMode::kTrain) {
    size_t n_rows = input.n_rows;
    size_t n_cols = input.n_cols;
    size_t n_slices = input.n_slices;
    arma::vec input_vec = DataTransformer::FlattenCube(input);
    arma::vec output_vec;
    Forward(input_vec, output_vec);
    output = DataTransformer::VecToCube(input_vec, n_rows, n_cols, n_slices);
  } else {
    output = input;
  }
}

void Dropout::Forward(const arma::vec& input, arma::vec& output,
                      const DropoutMode mode) {
  if (mode == DropoutMode::kTrain) {
    dropout_mask = arma::zeros(input.n_rows);
    dropout_mask = dropout_mask.imbue(
        [&]() { return RandomGenerator::GetUniformRandom(0, 1); });
    dropout_mask.transform([=](double val) { return val <= keep_prop ? 1 : 0; });
    dropout_mask /= keep_prop;
    output = input % dropout_mask;
  } else {
    output = input;
  }
}


void Dropout::Backward(const arma::vec& upstream_gradient) {
  grad_input = upstream_gradient % dropout_mask;
}

}  // namespace afs
