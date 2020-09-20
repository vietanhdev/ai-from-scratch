# AI From Scratch

**[WORK IN PROGRESS...]** This source code has been started recently, and It's still actively improved everyday. Feel free to help with any pull request.

**Motivation:** I started this project to build an artifical intelligence library from scratch for educational purpose. By that way, I want to understand clearly about AI algorithms (especially machine learning, deep learning) and practice C++ programming. I made this source code public with the hope that it can help others in understanding AI. I also look forward to your comments and suggestions to improve my source code. You can contact me directly using [this contact form](https://aicurious.io/contact).

## I. Suppported Layers

Supported layers (Documentation will be available soon).

- Fully connected
- Dropout
- ReLU
- Sigmoid
- Softmax
- Max pooling
- Convolutional (no padding)

## II. Examples

### 1. XOR Calculator

### 2. MNIST Digit Classification (using LeNet)

- Dataset: Download following dataset and extract `train.csv`, `test.csv` into `data/MNIST`.

https://www.kaggle.com/c/digit-recognizer/data

### 3. Wine Quality Estimator

- Dataset: Download following dataset and extract all files into `data/WineQuality`. You also need to open `winequality-red.csv` and replace all semicolon (`;`) with comma (`,`). This file will be used for Wine Quality Estimator example.

https://archive.ics.uci.edu/ml/datasets/wine+quality

## III. Setup and Run

### Environment

My source code was tested with Ubuntu 18.08. However, it also should work on other operating systems without problem.

- **Dependencies:** Armadillo (for matrix computation), OpenCV, CMake.

- Install `armadillo`:

```
sudo apt install libopenblas-dev liblapack-dev
wget http://sourceforge.net/projects/arma/files/armadillo-9.880.1.tar.xz
tar -xvf armadillo-9.880.1.tar.gz
cd armadillo-9.880.1
./configure
make
sudo make install
```

- Install OpenCV:

```
sudo apt intall libopencv-dev
```

### Compile and Run

- Compile:

```
mkdir build
cd build
cmake ..
make
```

- Run Wine Quality Estimator:

```
./wine_quality_estimator
```

## IV. References

- http://www.cs.virginia.edu/~vicente/vislang/notebooks/deep_learning_lab.html
- https://cs231n.github.io/optimization-2/