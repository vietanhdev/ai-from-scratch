#ifndef CROSS_ENTROPY_LOSS_H_
#define CROSS_ENTROPY_LOSS_H_

#include <armadillo>
#include <cassert>
#include <iostream>

class CrossEntropyLoss {
 private:
  size_t num_inputs;
  arma::vec predicted_distribution;
  arma::vec actual_distribution;

  double loss;

  arma::vec gradient_wrt_predicted_distribution;

 public:
  CrossEntropyLoss(size_t num_inputs) : num_inputs(num_inputs) {}

  double Forward(arma::vec& predicted_distribution,
                 arma::vec& actual_distribution) {
    assert(predicted_distribution.n_elem == num_inputs);
    assert(actual_distribution.n_elem == num_inputs);

    // Cache the prdicted and actual labels -- these will be required in the
    // backward pass.
    this->predicted_distribution = predicted_distribution;
    this->actual_distribution = actual_distribution;

    // Compute the loss and cache that too.
    this->loss =
        -arma::dot(actual_distribution, arma::log(predicted_distribution));
    return this->loss;
  }

  void Backward() {
    gradient_wrt_predicted_distribution =
        -(actual_distribution % (1 / predicted_distribution));
  }

  arma::vec GetGradientWrtPredictedDistribution() {
    return gradient_wrt_predicted_distribution;
  }
};

#endif