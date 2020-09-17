#ifndef WINE_QUALITY_H_
#define WINE_QUALITY_H_

#include <armadillo>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

class WineQualityData {
 public:
  WineQualityData(const std::string &data_file, double split_ratio) {
    assert(split_ratio <= 1 && split_ratio >= 0);

    arma::mat train_data_raw;

    train_data_raw.load(data_file, arma::csv_ascii);
    std::cout << train_data_raw.n_rows << " " << train_data_raw.n_cols
              << std::endl;

    std::vector<arma::vec> train_data_all;
    std::vector<arma::vec> train_labels_all;
    for (size_t idx = 0; idx < train_data_raw.n_rows; idx++) {
      arma::vec sample_vec = train_data_raw.row(idx).t();
      arma::vec features = sample_vec.subvec(0, sample_vec.n_rows - 2);
      features = arma::normalise(features);
      train_data_all.push_back(features);
      arma::vec labelvec =
          sample_vec.subvec(sample_vec.n_rows - 1, sample_vec.n_rows - 1);
      train_labels_all.push_back(labelvec);
    }

    int num_examples = train_data_all.size();

    // Shuffle the data
    std::vector<arma::vec> train_data_all_shuffled;
    std::vector<arma::vec> train_labels_all_shuffled;
    std::vector<int> indexes;
    indexes.reserve(train_data_all.size());
    for (int i = 0; i < train_data_all.size(); ++i) indexes.push_back(i);
    std::random_shuffle(indexes.begin(), indexes.end());
    for (int i = 0; i < train_data_all.size(); ++i) {
      train_data_all_shuffled.push_back(train_data_all[indexes[i]]);
      train_labels_all_shuffled.push_back(train_labels_all[indexes[i]]);
    }
    train_data_all = train_data_all_shuffled;
    train_labels_all = train_labels_all_shuffled;

    // Split train_data_all and train_labels_all into train and validation
    // parts.
    train_data = std::vector<arma::vec>(
        train_data_all.begin(),
        train_data_all.begin() + num_examples * split_ratio);
    train_labels = std::vector<arma::vec>(
        train_labels_all.begin(),
        train_labels_all.begin() + num_examples * split_ratio);

    validation_data = std::vector<arma::vec>(
        train_data_all.begin() + num_examples * split_ratio,
        train_data_all.end());
    validation_labels = std::vector<arma::vec>(
        train_labels_all.begin() + num_examples * split_ratio,
        train_labels_all.end());

    // TODO (vietanhdev): Add code to load test set
    // arma::mat test_data_raw;
    // test_data_raw.load(test_file, arma::csv_ascii);
    // test_data_raw = test_data_raw.submat(1, 0, test_data_raw.n_rows - 1,
    //                                      test_data_raw.n_cols - 1);
    // for (size_t idx = 0; idx < test_data_raw.n_rows; idx++) {
    //   arma::cube img(28, 28, 1, arma::fill::zeros);
    //   for (size_t r = 0; r < 28; r++)
    //     img.slice(0).row(r) =
    //         test_data_raw.row(idx).subvec(28 * r, 28 * r + 27);
    //   img.slice(0) /= 255.0;
    //   test_data.push_back(img);
    // }
  }

  std::vector<arma::vec> getTrainData() { return train_data; }

  std::vector<arma::vec> getValidationData() { return validation_data; }

  std::vector<arma::vec> getTestData() { return test_data; }

  std::vector<arma::vec> getTrainLabels() { return train_labels; }

  std::vector<arma::vec> getValidationLabels() { return validation_labels; }

 private:
  std::string data_dir;
  std::string train_file;
  std::string test_file;

  std::vector<arma::vec> train_data;
  std::vector<arma::vec> validation_data;
  std::vector<arma::vec> test_data;

  std::vector<arma::vec> train_labels;
  std::vector<arma::vec> validation_labels;
};

#endif