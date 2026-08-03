#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

cv::Rect getCentralROI(const cv::Mat& img);

cv::Mat GrabCut(
    const cv::Mat& frame,
    const cv::Rect& roi,
    const cv::Mat& motionMask
);

cv::Mat GrabCutMediaPipe(
    const cv::Mat& frame,
    const cv::Rect& roi
);

std::vector<cv::Mat> finalMask(
    const std::vector<cv::Mat>& frames,
    const std::vector<cv::Mat>& seeds,
    const std::string& sequenceName,
    bool useMediaPipeForSequence
);

cv::Mat GrabCutModifiedOnlyForShadow(
    const cv::Mat& frame,
     const cv::Rect& roi, 
     const cv::Mat& motionMask
);