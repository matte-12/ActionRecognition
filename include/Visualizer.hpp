#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class Visualizer {
public:
    void saveSequenceOutput(
        const std::vector<cv::Mat>& frames,
        const std::vector<cv::Rect>& boxes,
        int predictedLabel,
        const std::string& outputFolder
    );
};
