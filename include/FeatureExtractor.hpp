#pragma once

#include "Types.hpp"
#include <opencv2/opencv.hpp>
#include <vector>


class FeatureExtractor
{
public:
    SequenceFeatures extract(const SequenceData& sequence);

private:
    double computeMean(const std::vector<double>& values);
    double computeVariation(const std::vector<double>& values, double mean);
};

