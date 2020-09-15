#include <armadillo>
#include <iostream>

using namespace arma;
using namespace std;

int main() {
  mat A(2, 3);

  A << 1 << 2 << 3 << 3 << endr << 2 << 0 << 6 << 2 << endr << 3 << 4 << 9 << 7
    << endr;

  A.print("A:");

  mat B = orth(A);

  B.print("B:");

  cout << "The rank of A is " << arma::rank(A) << endl;

  return 0;
}