#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

void showMasks(const std::vector<cv::Mat>& masks);

void showBBoxes(
    const std::vector<cv::Mat>& frames,
    const std::vector<cv::Rect>& bboxes
);