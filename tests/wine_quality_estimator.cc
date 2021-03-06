#include <armadillo>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/plot.hpp>
#include <vector>

#include "datasets/wine_quality.h"
#include "layers/dense.h"
#include "layers/sigmoid.h"
#include "losses/mse_loss.h"
#include "utils/visualizer.h"
#include "utils/data_transformer.h"

using namespace afs;
using namespace std;

int main(int argc, char **argv) {
  // Load Wine quality data
  WineQualityData dataset("../data/WineQuality/winequality-red.csv", 0.8);

  std::vector<arma::vec> train_data = dataset.getTrainData();
  std::vector<arma::vec> train_labels = dataset.getTrainLabels();

  std::vector<arma::vec> validation_data = dataset.getValidationData();
  std::vector<arma::vec> validation_labels = dataset.getValidationLabels();

  assert(train_data.size() == train_labels.size());
  assert(validation_data.size() == validation_labels.size());

  std::vector<arma::vec> test_data = dataset.getTestData();

  std::cout << "Training data size: " << train_data.size() << std::endl;
  std::cout << "Validation data size: " << validation_data.size() << std::endl;
  std::cout << "Test data size: " << test_data.size() << std::endl;
  std::cout << std::endl;

  const size_t kTrainDataSize = train_data.size();
  const size_t kValidDataSize = validation_data.size();
  const double kLearningRate = 0.001;
  const size_t kEpochs = 16;
  const size_t kBatchSize = 32;
  const size_t kNumBatches = kTrainDataSize / kBatchSize;

  Dense d1(train_data[0].n_rows, 16);
  Sigmoid s1(16);
  Dense d2(16, 1);

  MSELoss l;

  // Initialize armadillo structures to store intermediate outputs (Ie. outputs
  // of hidden layers)
  arma::vec d1_out;
  arma::vec s1_out;
  arma::vec d2_out;

  // Initialize loss and cumulative loss. Cumulative loss totals loss over all
  // training examples in a minibatch.
  double loss;
  double epoch_loss = 0.0;
  double mini_batch_loss;

  std::vector<double> loss_history;
  std::vector<double> val_loss_history;

  for (size_t epoch = 0; epoch < kEpochs; ++epoch) {
    epoch_loss = 0.0;

    std::cout << "*** Epoch " << epoch + 1 << "/" << kEpochs << ":"
              << std::endl;

    for (size_t batch_idx = 0; batch_idx < kNumBatches; ++batch_idx) {
      mini_batch_loss = 0.0;
      for (size_t i = 0; i < kBatchSize; ++i) {
        // Forward pass
        d1.Forward(train_data[batch_idx * kBatchSize + i], d1_out);
        s1.Forward(d1_out, s1_out);
        d2.Forward(s1_out, d2_out);

        // Compute the loss
        loss = l.Forward(d2_out, train_labels[batch_idx * kBatchSize + i]);
        mini_batch_loss += loss;

        // Backward pass
        l.Backward();
        arma::vec grad_wrt_predicted_distribution =
            l.GetGradientWrtPredictedDistribution();
        d2.Backward(grad_wrt_predicted_distribution);
        arma::vec d2_grad = d2.GetGradientWrtInput();
        s1.Backward(d2_grad);
        arma::vec s1_grad = s1.GetGradientWrtInput();
        d1.Backward(s1_grad);
        arma::vec d1_grad = d1.GetGradientWrtInput();
      }
      epoch_loss += mini_batch_loss;
      loss_history.push_back(mini_batch_loss / kBatchSize);
      Visualizer::PlotGraph(loss_history, "Loss");

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
    for (size_t i = 0; i < kTrainDataSize; ++i) {
      // Forward pass
      d1.Forward(train_data[i], d1_out);
      s1.Forward(d1_out, s1_out);
      d2.Forward(s1_out, d2_out);

      if ((int)train_labels[i][0] == (int)(round(d2_out[0]))) {
        correct += 1.0;
      }
    }

    // Output accuracy on training dataset after each epoch
    std::cout << "Training accuracy: " << correct / kTrainDataSize << std::endl;

    // Compute validation accuracy after epoch
    epoch_loss = 0.0;
    correct = 0.0;
    for (size_t i = 0; i < kValidDataSize; ++i) {
      // Forward pass
      d1.Forward(validation_data[i], d1_out);
      s1.Forward(d1_out, s1_out);
      d2.Forward(s1_out, d2_out);

      // Compute the loss
      loss = l.Forward(d2_out, validation_labels[i]);
      epoch_loss += loss;

      if ((int)train_labels[i][0] == (int)(round(d2_out[0]))) {
        correct += 1.0;
      }
    }

    // Output validation loss after each epoch
    double avg_loss_on_sample = epoch_loss / (kBatchSize * kNumBatches);
    val_loss_history.push_back(avg_loss_on_sample);
    Visualizer::PlotGraph(val_loss_history, "Val Loss");
    std::cout << "Validation loss: " << avg_loss_on_sample
              << std::endl;

    // Output validation accuracy after each epoch
    std::cout << "Val accuracy: " << correct / kValidDataSize << std::endl;
    std::cout << std::endl;
  }

  cv::waitKey(0);
}
