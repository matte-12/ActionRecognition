#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "Types.hpp"

class Visualizer {
public:
    void saveSequenceOutput(
        const SequenceData& sequenza,
        int predictedLabel,
        const std::string& outputFolder
    );
};
