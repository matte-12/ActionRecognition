#pragma once

#include "Types.hpp"

#include <opencv2/ml.hpp>
#include <vector>

class Classifier
{
public:
    Classifier();

    void train(const std::vector<SequenceFeatures>& trainingFeatures);

    int predict(const SequenceFeatures& features) const;

private:
    cv::Ptr<cv::ml::RTrees> staticForest;
    cv::Ptr<cv::ml::RTrees> dynamicForest;

    bool trained = false;
};
