#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct SequenceData {
    std::string name;
    std::vector<cv::Mat> frames;
    int trueLabel = -1; // label del tipo di azione: 1= WALKING  2= ... -->  cariate dalla GT in input
    cv::Rect gtBBox20; // nella groundTruth.txt ho i valori x_medio, y_medio. Li converto in x, y del BBox
    bool hasGT= false; // mi dice se è presente la groundtrouth.txt
    
    std::vector<cv::Rect> BBoxes; // contiene tutti i boundig box della sequenza
    std::vector<cv::Mat> finalMasks; // contiene tutte le mask risultanti della sequenza
    std::vector<cv::Mat> motionMasks; // puo servire per featureExtraction
};

struct SequenceFeatures
{
    std::string sequenceName;
    int trueLabel = -1;

    // Geometria bbox
    double meanArea = 0.0;
    double areaVariation = 0.0;

    double meanAspectRatio = 0.0;
    double aspectRatioVariation = 0.0;

    double meanBoxWidth = 0.0;
    double meanBoxHeight = 0.0;
    double widthVariation = 0.0;
    double widthVariationNorm = 0.0;

    // Traiettoria
    double totalDisplacementX = 0.0;
    double totalDisplacementY = 0.0;
    double totalDisplacement = 0.0;

    double displacementNorm = 0.0;

    double pathLength = 0.0;
    double pathLengthNorm = 0.0;

    double meanVelocity = 0.0;
    double velocityNorm = 0.0;

    double trajectoryEfficiency = 0.0;

    // Movimento
    double motionEnergySeed = 0.0;

    double motionEnergyFinalMask = 0.0;

    double upperMotion = 0.0;
    double middleMotion = 0.0;
    double lowerMotion = 0.0;

    double leftMotion = 0.0;
    double rightMotion = 0.0;
    double leftRightDifference = 0.0;

    double centerMotionRatio = 0.0;

    // optical flow
    // Intensità media del movimento
    double meanFlowMagnitude = 0.0;

    // Picco massimo di movimento nella sequenza
    double maxFlowMagnitude = 0.0;

    // Movimento orizzontale nella parte superiore del corpo
    double upperHorizontalFlow = 0.0;

    // Movimento verticale nella parte superiore del corpo
    double upperVerticalFlow = 0.0;

    // Quanto il movimento tende a convergere/divergere verso/da il centro del corpo
    double flowConvergence = 0.0;

    // Intensità del movimento nella parte inferiore del corpo
    double lowerFlowMagnitude = 0.0;
    // simmetria dell'azione
    double leftX = 0.0;
    double rightX = 0.0;

    double leftAbsX = 0.0;
    double rightAbsX = 0.0;

    double horizontalSymmetry = 0.0;

    // utile per jogging vs running
    double strideFrequency = 0.0;
};
