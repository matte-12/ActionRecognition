#pragma once

#include <opencv2/opencv.hpp>

cv::Mat createDarkMask(const cv::Mat& frame);

bool isShadowOrAbruptChangeFrame(
    const cv::Rect& currentBox,
    const cv::Rect& previousGoodBox,
    const cv::Mat& currentMask,
    const cv::Size& frameSize
);

bool shouldAcceptModifiedShadowResult(
    const cv::Rect& originalBox,
    const cv::Rect& modifiedBox,
    const cv::Rect& previousGoodBox,
    const cv::Size& frameSize
);

std::vector<cv::Mat> correctOnlyShadowFramesWithModifiedSegmentation(
    const std::vector<cv::Mat>& frames,
    const std::vector<cv::Mat>& seeds,
    const std::vector<cv::Mat>& originalMasks,
    std::vector<cv::Rect>& bboxes,
    const std::string& sequenceName
);