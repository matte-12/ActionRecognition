#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct SequenceData {
    std::string name;
    std::vector<cv::Mat> frames;
    //int trueLabel = -1;
    //cv::Rect gtBBoxFrame20;
    
};

struct PredictionResult {
    std::string sequenceName;
    int trueLabel = -1;
    int predictedLabel = -1;
    cv::Rect predictedBoxFrame20;
    cv::Rect groundTruthBoxFrame20;
    double iou = 0.0;
};
