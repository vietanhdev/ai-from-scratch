#ifndef VISUALIZER_H_
#define VISUALIZER_H_

#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/plot.hpp>
#include <string>
#include <vector>

class Visualizer {
 public:
  static void PlotGraph(const std::vector<double> &input_data,
                        const std::string &title) {
    // Create Ploter
    cv::Mat data(input_data);
    cv::transpose(data, data);
    cv::Ptr<cv::plot::Plot2d> plot = cv::plot::Plot2d::create(data);
    plot->setInvertOrientation(true);

    // Render Plot Image
    cv::Mat image;
    plot->render(image);

    // Show Image
    cv::imshow(title, image);
    cv::waitKey(1);
  }
};

#endif