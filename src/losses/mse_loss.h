#ifndef MSE_LOSS_H_
#define MSE_LOSS_H_

#include <armadillo>
#include <cassert>
#include <iostream>

class MSELoss {
 private:
  arma::vec predicted_distribution;
  arma::vec actual_distribution;

  double loss;

  arma::vec gradient_wrt_predicted_distribution;

 public:

  double Forward(arma::vec& predicted_distribution,
                 arma::vec& actual_distribution) {

    // Cache the prdicted and actual labels -- these will be required in the
    // backward pass.
    this->predicted_distribution = predicted_distribution;
    this->actual_distribution = actual_distribution;

    // Compute the loss and cache that too.
    this->loss = arma::accu(square(actual_distribution - predicted_distribution));
    return this->loss;
  }

  void Backward() {
    int num_samples = this->predicted_distribution.n_rows;
    gradient_wrt_predicted_distribution = num_samples * 2 * (predicted_distribution - actual_distribution);
  }

  arma::vec GetGradientWrtPredictedDistribution() {
    return gradient_wrt_predicted_distribution;
  }
};

#endif