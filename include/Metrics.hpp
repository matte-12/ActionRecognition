#pragma once

#include <opencv2/opencv.hpp>
#include "Types.hpp"

class Metrics {
public:
    double computeIoU(const cv::Rect& a, const cv::Rect& b);
};
