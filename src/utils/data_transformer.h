#ifndef DATA_TRANSFORMER_H_
#define DATA_TRANSFORMER_H_

#include <armadillo>
#include <iostream>

namespace afs {

class DataTransformer {
 public:
  static arma::cube VecToCube(arma::vec vec_in, size_t n_rows, size_t n_cols, size_t n_slices) {
    arma::cube tmp((n_rows * n_cols * n_slices), 1, 1);
    tmp.slice(0).col(0) = vec_in;
    arma::cube cube_out = arma::reshape(tmp, n_rows, n_cols, n_slices);
    return cube_out;
  }

  static arma::vec FlattenCube(arma::cube cube_in) {
    cube_in = arma::reshape(cube_in, cube_in.n_rows * cube_in.n_cols * cube_in.n_slices, 1, 1);
    arma::vec flattened = arma::vectorise(cube_in);
    return flattened;
  }
};

}  // namespace afs

#endif