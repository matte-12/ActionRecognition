#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct SequenceData {
    std::string name;
    std::vector<cv::Mat> frames;
    int trueLabel = -1; // label del tipo di azione: 1= WALKING  2= ... --> ricavate con FeatureExtraction
    cv::Rect gtBBox20; // nella groundTruth.txt ho i valori x_medio, y_medio. Li converto in x, y del BBox
    bool hasGT= false; // mi dice se è presente la groundtrouth.txt
    
    std::vector<cv::Rect> BBoxes; // contiene tutti i boundig box della sequenza
    std::vector<cv::Mat> finalMasks; // contiene tutte le mask risultanti della sequenza
    std::vector<cv::Mat> motionMasks; // puo servire per featureExtraction
};

struct SequenceFeatures
{
    std::string sequenceName;

    double meanArea = 0.0;
    double areaVariation = 0.0;

    double meanAspectRatio = 0.0;

    double totalDisplacementX = 0.0;
    double totalDisplacementY = 0.0;
    double totalDisplacement = 0.0;

    double meanVelocity = 0.0;

    double motionEnergy = 0.0;

    double upperMotion = 0.0;
    double middleMotion = 0.0;
    double lowerMotion = 0.0;

    int trueLabel = -1;
};

struct Prediction {
    std::string sequenceName;
    int trueLabel = -1;
    int predictedLabel = -1;
    cv::Rect predictedBoxFrame20;
    cv::Rect groundTruthBoxFrame20;
    double iou = 0.0;
};
