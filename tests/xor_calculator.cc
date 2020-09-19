#include <armadillo>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

#include "layers/dense.h"
#include "layers/sigmoid.h"
#include "utils/data_transformer.h"
#include "losses/mse_loss.h"

using namespace afs;
using namespace std;

int main(int argc, char **argv) {
  std::vector<arma::vec> train_data;
  std::vector<arma::vec> train_labels;

  for (double i = 0; i <= 1; ++i) {
    for (double j = 0; j <= 1; ++j) {
      arma::vec data({i, j});
      arma::vec label({(double)((int)i ^ (int)j)});
      train_data.push_back(data);
      train_labels.push_back(label);
    }
  }

  const size_t kTrainDataSize = train_data.size();
  const double kLearningRate = 1.0;
  const size_t kEpochs = 5000;
  const size_t kBatchSize = 1;
  const size_t kNumBatches = kTrainDataSize / kBatchSize;

  Dense d1(2, 4);
  Sigmoid s1(4);
  Dense d2(4, 1);
  Sigmoid s2(1);

  MSELoss l;

  // Initialize armadillo structures to store intermediate outputs (Ie. outputs
  // of hidden layers)
  arma::vec d1_out = arma::zeros(4);
  arma::vec s1_out = arma::zeros(4);
  arma::vec d2_out = arma::zeros(1);
  arma::vec s2_out = arma::zeros(1);

  // Initialize loss and cumulative loss. Cumulative loss totals loss over all
  // training examples in a minibatch.
  double loss;
  double epoch_loss = 0.0;
  double mini_batch_loss;

  for (size_t epoch = 0; epoch < kEpochs; epoch++) {
    std::cout << "*** Epoch " << epoch + 1 << "/" << kEpochs << ":"
              << std::endl;

    for (size_t batch_idx = 0; batch_idx < kNumBatches; batch_idx++) {
      mini_batch_loss = 0.0;
      for (size_t i = 0; i < kBatchSize; i++) {
        // Forward pass
        d1.Forward(train_data[batch_idx * kBatchSize + i], d1_out);
        s1.Forward(d1_out, s1_out);
        d2.Forward(s1_out, d2_out);
        s2.Forward(d2_out, s2_out);

        // Compute the loss
        loss = l.Forward(s2_out, train_labels[batch_idx * kBatchSize + i]);
        mini_batch_loss += loss;

        // Backward pass
        l.Backward();
        arma::vec grad_wrt_predicted_distribution =
            l.GetGradientWrtPredictedDistribution();
        s2.Backward(grad_wrt_predicted_distribution);
        arma::vec s2_grad = s2.GetGradientWrtInput();
        d2.Backward(s2_grad);
        arma::vec d2_grad = d2.GetGradientWrtInput();
        s1.Backward(d2_grad);
        arma::vec s1_grad = s1.GetGradientWrtInput();
        d1.Backward(s1_grad);
        arma::vec d1_grad = d1.GetGradientWrtInput();
      }
      epoch_loss += mini_batch_loss;

      std::cout << '\r' << "Batch " << batch_idx + 1 << "/" << kNumBatches
                << " Batch loss: " << mini_batch_loss << std::flush;

      // Update params
      d1.UpdateWeightsAndBiases(kBatchSize, kLearningRate);
      d2.UpdateWeightsAndBiases(kBatchSize, kLearningRate);
    }

    // Output loss on training dataset after each epoch
    std::cout << std::endl;
    std::cout << "Training loss: " << epoch_loss / (kBatchSize * kNumBatches)
              << std::endl;

    // Compute the training accuracy after epoch
    double correct = 0.0;
    for (size_t i = 0; i < kTrainDataSize; i++) {
      // Forward pass
      d1.Forward(train_data[i], d1_out);
      s1.Forward(d1_out, s1_out);
      d2.Forward(s1_out, d2_out);
      s2.Forward(d2_out, s2_out);
    
      cout << (int)train_data[i][0]
            << " XOR " << (int)train_data[i][1]
            << " = " << s2_out[0] << " ~ " << (int)train_labels[i][0]
            << endl;

      if ((int)train_labels[i][0] == (int)(s2_out[0] > 0.5)) {
        correct += 1.0;
      }

    }

    // Output accuracy on training dataset after each epoch
    std::cout << "Training accuracy: " << correct / kTrainDataSize << std::endl;

    // Reset cumulative loss and correct count
    epoch_loss = 0.0;
    correct = 0.0;
  }
}
