#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

std::vector<cv::Mat> fillEmptySeedsTemporal(
    const std::vector<cv::Mat>& seeds,
    int minNonZero = 10
);

std::vector<cv::Mat> filterMorphology(
    const std::vector<cv::Mat>& frames
);

bool isLightChangeMask(
    const cv::Mat& mask,
    double threshold = 0.35
);


bool shouldUseMediaPipeForSequence(
    const std::vector<cv::Mat>& seeds,
    double zeroRatioThreshold = 0.80
);

std::vector<cv::Mat> filterMOG2(
    const std::vector<cv::Mat>& frames
);

cv::Mat createCleanForegroundSeedModified(
    const cv::Mat& frame,
    const cv::Mat& motionMask
);

// da completare forse