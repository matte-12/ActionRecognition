#pragma once

#include "Types.hpp"
#include <string>
#include <vector>
#include <filesystem>

class DatasetLoader {
public:
    std::vector<SequenceData> loadDataset(const std::string& datasetPath);

private: 
    std::vector<std::string> getSubfolderPath(const std::string& datasetPath);
    std::vector<cv::Mat> getFrames(const std::string& datasetPath);
    bool getGroundTruth ( SequenceData& sequence );
};
