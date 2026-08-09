#pragma once

#include <opencv2/opencv.hpp>
#include <vector>


std::vector<cv::Rect> smoothBBoxes(
    const std::vector<cv::Rect>& bboxes,
    double alpha = 0.35,
    int maxEmptyGap = 3
);

std::vector<cv::Rect> findBBox(
    const std::vector<cv::Mat>& masks,
    int minContourArea = 1
);
