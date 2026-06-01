#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

class Segmentation {
public:
    std::vector<cv::Rect> extractBBox(const std::vector<cv::Mat>& frames);

private: 
};
