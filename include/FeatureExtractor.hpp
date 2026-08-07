#pragma once

#include "Types.hpp"
#include <opencv2/opencv.hpp>
#include <vector>


class FeatureExtractor
{
public:
    SequenceFeatures extract(const SequenceData& sequence);
    
    void logFeatures(
        const SequenceFeatures& features,
        const std::string& logPath);
};

