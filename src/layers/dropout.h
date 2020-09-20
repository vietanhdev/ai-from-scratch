#ifndef DROPOUT_H_
#define DROPOUT_H_

#include <armadillo>
#include <cassert>
#include <cmath>
#include <vector>

namespace afs {

enum class DropoutMode {kTrain, kTest};

class Dropout {
 public:

  Dropout(float keep_prop = 1.0);

  void Forward(const arma::vec& input, arma::vec& output, const DropoutMode mode = DropoutMode::kTrain);
  void Forward(const arma::cube& input, arma::cube& output, const DropoutMode mode = DropoutMode::kTrain);
  void Backward(const arma::vec& upstream_gradient);
  arma::vec GetGradientWrtInput() { return grad_input; }

 private:
  float keep_prop;
  arma::vec dropout_mask;
  arma::vec grad_input;

  void ResetGradient();
};

}  // namespace afs

#endif