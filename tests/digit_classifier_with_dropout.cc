#include <armadillo>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

#include "datasets/mnist.h"
#include "layers/conv2d.h"
#include "layers/dense.h"
#include "layers/max_pooling.h"
#include "layers/relu.h"
#include "layers/softmax.h"
#include "layers/dropout.h"
#include "losses/cross_entropy_loss.h"
#include "utils/visualizer.h"
#include "utils/data_transformer.h"

using namespace afs;
using namespace std;

int main(int argc, char **argv) {
  // Load MNIST data
  MNISTData md("../data/MNIST");

  std::vector<arma::cube> train_data = md.getTrainData();
  std::vector<arma::vec> train_labels = md.getTrainLabels();

  std::vector<arma::cube> validation_data = md.getValidationData();
  std::vector<arma::vec> validation_labels = md.getValidationLabels();

  assert(train_data.size() == train_labels.size());
  assert(validation_data.size() == validation_labels.size());

  std::vector<arma::cube> test_data = md.getTestData();

  std::cout << "Training data size: " << train_data.size() << std::endl;
  std::cout << "Validation data size: " << validation_data.size() << std::endl;
  std::cout << "Test data size: " << test_data.size() << std::endl;
  std::cout << std::endl;

  const size_t kTrainDataSize = train_data.size();
  const size_t kValidDataSize = validation_data.size();
  const size_t kTestDataSize = test_data.size();
  const double kLearningRate = 0.01;
  const size_t kEpochs = 10;
  const size_t kBatchSize = 16;
  const size_t kNumBatches = kTrainDataSize / kBatchSize;

  // Define the network layers
  Conv2D c1(28, 28, 1, 5, 5, 1, 1, 6);
  // Output is 24 x 24 x 6

  ReLU r1(24, 24, 6);
  // Output is 24 x 24 x 6

  MaxPooling mp1(24, 24, 6, 2, 2, 2, 2);
  // Output is 12 x 12 x 6

  Conv2D c2(12, 12, 6, 5, 5, 1, 1, 16);
  // Output is 8 x 8 x 16

  Dropout c2_dropout(0.5);

  ReLU r2(8, 8, 16);
  // Output is 8 x 8 x 16

  MaxPooling mp2(8, 8, 16, 2, 2, 2, 2);
  // Output is 4 x 4 x 16

  Dense d(4 * 4 * 16, 10);
  // Output is a vector of size 10

  Softmax s(10);
  // Output is a vector of size 10

  CrossEntropyLoss l(10);

  // Initialize armadillo structures to store intermediate outputs (Ie. outputs
  // of hidden layers)
  arma::cube c1_out = arma::zeros(24, 24, 6);
  arma::cube r1_out = arma::zeros(24, 24, 6);
  arma::cube mp1_out = arma::zeros(12, 12, 6);
  arma::cube c2_out = arma::zeros(8, 8, 16);
  arma::cube c2_dropout_out;
  arma::cube r2_out = arma::zeros(8, 8, 16);
  arma::cube mp2_out = arma::zeros(4, 4, 16);
  arma::vec d_out = arma::zeros(10);
  arma::vec s_out = arma::zeros(10);

  // Initialize loss and cumulative loss. Cumulative loss totals loss over all
  // training examples in a minibatch.
  double loss;
  double epoch_loss = 0.0;
  double mini_batch_loss;
  std::vector<double> train_loss_history;

  for (size_t epoch = 0; epoch < kEpochs; ++epoch) {
    std::cout << "*** Epoch " << epoch + 1 << "/" << kEpochs << ":"
              << std::endl;

    for (size_t batch_idx = 0; batch_idx < kNumBatches; ++batch_idx) {
      mini_batch_loss = 0.0;
      for (size_t i = 0; i < kBatchSize; ++i) {
        // Forward pass
        c1.Forward(train_data[batch_idx * kBatchSize + i], c1_out);
        r1.Forward(c1_out, r1_out);
        mp1.Forward(r1_out, mp1_out);
        c2.Forward(mp1_out, c2_out);
        c2_dropout.Forward(c2_out, c2_dropout_out);
        r2.Forward(c2_dropout_out, r2_out);
        mp2.Forward(r2_out, mp2_out);
        d.Forward(mp2_out, d_out);
        s.Forward(d_out, s_out);

        // Compute the loss
        loss = l.Forward(s_out, train_labels[batch_idx * kBatchSize + i]);
        mini_batch_loss += loss;

        // Backward pass
        l.Backward();
        arma::vec grad_wrt_predicted_distribution =
            l.GetGradientWrtPredictedDistribution();
        s.Backward(grad_wrt_predicted_distribution);
        arma::vec grad_wrt_s_in = s.GetGradientWrtInput();
        d.Backward(grad_wrt_s_in);
        arma::vec grad_wrt_d_in_vec = d.GetGradientWrtInput();
        arma::cube grad_wrt_d_in =
            DataTransformer::VecToCube(grad_wrt_d_in_vec, mp2.output.n_rows,
                             mp2.output.n_cols, mp2.output.n_slices);
        mp2.Backward(grad_wrt_d_in);
        arma::cube grad_wrt_mp2_in = mp2.GetGradientWrtInput();
        r2.Backward(grad_wrt_mp2_in);
        arma::cube grad_wrt_r2_in = r2.GetGradientWrtInput();
        arma::cube grad_wrt_c2_dropout_in = c2_dropout.Backward(grad_wrt_r2_in);
        c2.Backward(grad_wrt_c2_dropout_in);
        arma::cube grad_wrt_c2_in = c2.GetGradientWrtInput();
        mp1.Backward(grad_wrt_c2_in);
        arma::cube grad_wrt_mp1_in = mp1.GetGradientWrtInput();
        r1.Backward(grad_wrt_mp1_in);
        arma::cube grad_wrt_r1_in = r1.GetGradientWrtInput();
        c1.Backward(grad_wrt_r1_in);
        arma::cube grad_wrt_c1_in = c1.GetGradientWrtInput();
      }
      epoch_loss += mini_batch_loss;

      train_loss_history.push_back(mini_batch_loss / kBatchSize);
      Visualizer::PlotGraph(train_loss_history, "Loss");

      std::cout << '\r' << "Batch " << batch_idx + 1 << "/" << kNumBatches
                << " Batch loss: " << mini_batch_loss << std::flush;

      // Update params
      d.UpdateWeightsAndBiases(kBatchSize, kLearningRate);
      c1.UpdateFilterWeights(kBatchSize, kLearningRate);
      c2.UpdateFilterWeights(kBatchSize, kLearningRate);
    }

    // Output loss on training dataset after each epoch
    std::cout << std::endl;
    std::cout << "Training loss: " << epoch_loss / (kBatchSize * kNumBatches)
              << std::endl;

    // Compute the training accuracy after epoch
    double correct = 0.0;
    for (size_t i = 0; i < kTrainDataSize; ++i) {
      // Forward pass
      c1.Forward(train_data[i], c1_out);
      r1.Forward(c1_out, r1_out);
      mp1.Forward(r1_out, mp1_out);
      c2.Forward(mp1_out, c2_out);
      c2_dropout.Forward(c2_out, c2_dropout_out, DropoutMode::kTest);
      r2.Forward(c2_dropout_out, r2_out);
      mp2.Forward(r2_out, mp2_out);
      d.Forward(mp2_out, d_out);
      s.Forward(d_out, s_out);

      if (train_labels[i].index_max() == s_out.index_max()) correct += 1.0;
    }

    // Output accuracy on training dataset after each epoch
    std::cout << "Training accuracy: " << correct / kTrainDataSize << std::endl;

    // Compute validation accuracy after epoch
    epoch_loss = 0.0;
    correct = 0.0;
    for (size_t i = 0; i < kValidDataSize; ++i) {
      // Forward pass
      c1.Forward(validation_data[i], c1_out);
      r1.Forward(c1_out, r1_out);
      mp1.Forward(r1_out, mp1_out);
      c2.Forward(mp1_out, c2_out);
      c2_dropout.Forward(c2_out, c2_dropout_out, DropoutMode::kTest);
      r2.Forward(c2_dropout_out, r2_out);
      mp2.Forward(r2_out, mp2_out);
      d.Forward(mp2_out, d_out);
      s.Forward(d_out, s_out);

      epoch_loss += l.Forward(s_out, validation_labels[i]);

      if (validation_labels[i].index_max() == s_out.index_max()) correct += 1.0;
    }

    // Output validation loss after each epoch
    std::cout << "Validation loss: " << epoch_loss / (kBatchSize * kNumBatches)
              << std::endl;

    // Output validation accuracy after each epoch
    std::cout << "Val accuracy: " << correct / kValidDataSize << std::endl;
    std::cout << std::endl;

    // Reset cumulative loss and correct count
    epoch_loss = 0.0;
    correct = 0.0;

    // Write results on test data to results csv
    std::fstream fout("lenet_" + std::to_string(epoch) + ".csv",
                      std::ios::out);
    fout << "ImageId,Label" << std::endl;
    for (size_t i = 0; i < kTestDataSize; ++i) {
      // Forward pass
      c1.Forward(test_data[i], c1_out);
      r1.Forward(c1_out, r1_out);
      mp1.Forward(r1_out, mp1_out);
      c2.Forward(mp1_out, c2_out);
      c2_dropout.Forward(c2_out, c2_dropout_out, DropoutMode::kTest);
      r2.Forward(c2_dropout_out, r2_out);
      mp2.Forward(r2_out, mp2_out);
      d.Forward(mp2_out, d_out);
      s.Forward(d_out, s_out);

      fout << std::to_string(i + 1) << "," << std::to_string(s_out.index_max())
            << std::endl;
    }
    fout.close();
  }

  cv::waitKey(0);

}
