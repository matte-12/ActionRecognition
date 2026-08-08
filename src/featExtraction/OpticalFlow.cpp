#include "featExtraction/OpticalFlow.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>


// trova stride: quante oscillazioni/picchi
// evito fft per ora. Usa trick: il valore deve essere sopra la media e maggiore dei vicini
static double computeStrideFrequency(
    const std::vector<double>& lowerMotionSignal)
{
    if (lowerMotionSignal.size() < 3)
        return 0.0;

    double mean = 0.0;

    for (double value : lowerMotionSignal)
        mean += value;

    mean /= static_cast<double>(lowerMotionSignal.size());

    int peaks = 0;

    for (size_t i = 1; i < lowerMotionSignal.size() - 1; i++)
    {
        bool localMaximum =
            lowerMotionSignal[i] > lowerMotionSignal[i - 1] && 
            lowerMotionSignal[i] > lowerMotionSignal[i + 1];
        // evito piccoli picchi di rumore
        bool significantPeak = lowerMotionSignal[i] > mean * 1.10 ;

        if (localMaximum && significantPeak)
            peaks++;
    }

    return static_cast<double>(peaks) / static_cast<double>(lowerMotionSignal.size());
}

// ------------------------------------------------------------
// converte un frame in grayscale


static cv::Mat toGray(const cv::Mat& frame)
{
    if (frame.empty())
        return cv::Mat();

    if (frame.channels() == 1)
        return frame;

    cv::Mat gray;

    cv::cvtColor(
        frame,
        gray,
        cv::COLOR_BGR2GRAY
    );

    return gray;
}


// ------------------------------------------------------------
// calcola una bbox sicura dentro i limiti immagine


static cv::Rect getSafeBox(
    const cv::Rect& box,
    const cv::Size& imageSize)
{
    cv::Rect imageRect(
        0,
        0,
        imageSize.width,
        imageSize.height
    );

    return box & imageRect;
}


// ------------------------------------------------------------
//  calcola media della magnitude in una ROI


static double computeMeanMagnitude(
    const cv::Mat& magnitude,
    const cv::Rect& roi)
{
    if (roi.width <= 0 || roi.height <= 0)
        return 0.0;

    return cv::mean(magnitude(roi))[0];
}



// ------------------------------------------------------------
// convergenza orizzontale
//
// metà sinistra:
// flow X > 0 = verso il centro
//
// metà destra:
// flow X < 0 = verso il centro


static double computeFlowConvergence(
    const cv::Mat& flowX,
    const cv::Rect& box)
{
    if (box.width < 2 || box.height <= 0)
        return 0.0;

    int halfWidth = box.width / 2;

    cv::Rect leftROI(
        box.x,
        box.y,
        halfWidth,
        box.height
    );

    cv::Rect rightROI(
        box.x + halfWidth,
        box.y,
        box.width - halfWidth,
        box.height
    );

    cv::Scalar leftMean = cv::mean(flowX(leftROI));

    cv::Scalar rightMean = cv::mean(flowX(rightROI));

    double leftX = leftMean[0];

    double rightX = rightMean[0];

    double convergence = 0.0;

    if (leftX > 0.0)
        convergence += leftX;

    if (rightX < 0.0)
        convergence += std::abs(rightX);

    return convergence;
}


// ------------------------------------------------------------
// Funzione principale


OpticalFlow computeOpticalFlow( const SequenceData& sequence)
{
    OpticalFlow features;

    const std::vector<cv::Mat>& frames = sequence.frames;

    const std::vector<cv::Rect>& boxes = sequence.BBoxes;

    if (frames.size() != boxes.size())
    {
        std::cerr
            << "[OpticalFlow] SIZE MISMATCH: frames="
            << frames.size()
            << " boxes="
            << boxes.size()
            << std::endl;

        return features;
    }

    if (frames.size() < 2)
        return features;

    double totalMeanMagnitude = 0.0;

    double totalUpperHorizontal = 0.0;
    double totalUpperVertical = 0.0;

    double totalLowerMagnitude = 0.0;

    double totalConvergence = 0.0;

    double globalMaxMagnitude = 0.0;

    double totalLeftX = 0.0;
    double totalRightX = 0.0;

    double totalLeftAbsX = 0.0;
    double totalRightAbsX = 0.0;

    int validPairs = 0;

    // utile poi per stride frequency
    std::vector<double> lowerMotionSignal;

    for (size_t i = 1; i < frames.size(); i++)
    {
        if (frames[i - 1].empty() ||
            frames[i].empty())
        {
            continue;
        }

        if (boxes[i - 1].area() <= 0 ||
            boxes[i].area() <= 0)
        {
            continue;
        }

        cv::Mat previousGray = toGray(frames[i - 1]);

        cv::Mat currentGray = toGray(frames[i]);

        if (previousGray.empty() || currentGray.empty())
        {
            continue;
        }

        // Dense Optical Flow Farnebäck

        cv::Mat flow;

        cv::calcOpticalFlowFarneback(
            previousGray,
            currentGray,
            flow,
            0.5,   // pyr_scale
            3,     // levels
            15,    // winsize
            3,     // iterations
            5,     // poly_n
            1.2,   // poly_sigma
            0      // flags
        );

        // bbox del frame corrente
        cv::Rect safeBox = getSafeBox(boxes[i], flow.size());

        if (safeBox.width < 4 ||
            safeBox.height < 4)
        {
            continue;
        }

        // separa flow X e flow Y

        std::vector<cv::Mat> flowChannels;

        cv::split(flow, flowChannels);

        cv::Mat& flowX = flowChannels[0];

        cv::Mat& flowY = flowChannels[1];

        // magnitude

        cv::Mat magnitude;
        cv::Mat angle;

        cv::cartToPolar(
            flowX,
            flowY,
            magnitude,
            angle,
            false
        );

        double meanMagnitude =
            computeMeanMagnitude(
                magnitude,
                safeBox
            );

        totalMeanMagnitude +=
            meanMagnitude;

        double minValue = 0.0;
        double maxValue = 0.0;

        cv::minMaxLoc(
            magnitude(safeBox),
            &minValue,
            &maxValue
        );

        if (maxValue > globalMaxMagnitude)
        {
            globalMaxMagnitude =
                maxValue;
        }

    
        // Upper / Lower body
        

        int halfHeight =
            safeBox.height / 2;

        cv::Rect upperBox(
            safeBox.x,
            safeBox.y,
            safeBox.width,
            halfHeight
        );

        cv::Rect lowerBox(
            safeBox.x,
            safeBox.y + halfHeight,
            safeBox.width,
            safeBox.height - halfHeight
        );

        int halfWidth = safeBox.width / 2;

        cv::Rect leftROI(
            safeBox.x,
            safeBox.y,
            halfWidth,
            safeBox.height
        );

        cv::Rect rightROI(
            safeBox.x + halfWidth,
            safeBox.y,
            safeBox.width - halfWidth,
            safeBox.height
        );

        // right / left horizontal motion
        double leftX = cv::mean(flowX(leftROI))[0];

        double rightX = cv::mean(flowX(rightROI))[0];

        cv::Mat leftAbs;
        cv::Mat rightAbs;

        cv::absdiff(
            flowX(leftROI),
            cv::Scalar(0),
            leftAbs
        );

        cv::absdiff(
            flowX(rightROI),
            cv::Scalar(0),
            rightAbs
        );

        double leftAbsX = cv::mean(leftAbs)[0];

        double rightAbsX = cv::mean(rightAbs)[0];

        totalLeftX += leftX;
        totalRightX += rightX;

        totalLeftAbsX += leftAbsX;
        totalRightAbsX += rightAbsX;

        // Upper horizontal motion
        cv::Mat upperFlowX =
            flowX(upperBox);

        cv::Mat upperFlowY =
            flowY(upperBox);

        cv::Mat absUpperX;
        cv::Mat absUpperY;

        cv::absdiff(
            upperFlowX,
            cv::Scalar(0),
            absUpperX
        );

        cv::absdiff(
            upperFlowY,
            cv::Scalar(0),
            absUpperY
        );

        totalUpperHorizontal += cv::mean(absUpperX)[0];

        totalUpperVertical += cv::mean(absUpperY)[0];

        // Lower body magnitude

        double lowerMagnitude =
            computeMeanMagnitude(
                magnitude,
                lowerBox
            );

        totalLowerMagnitude += lowerMagnitude;

        lowerMotionSignal.push_back(lowerMagnitude);


        // Flow convergence

        double convergence =
            computeFlowConvergence(
                flowX,
                safeBox
            );

        totalConvergence += convergence;

        validPairs++;
    }

    if (validPairs == 0)
        return features;


    // Feature finali

    features.leftX = totalLeftX / static_cast<double>(validPairs);

    features.rightX = totalRightX / static_cast<double>(validPairs);

    features.leftAbsX = totalLeftAbsX / static_cast<double>(validPairs);

    features.rightAbsX = totalRightAbsX / static_cast<double>(validPairs);

    features.horizontalSymmetry =
    std::abs(
        features.leftAbsX -
        features.rightAbsX
    );

    features.meanFlowMagnitude = totalMeanMagnitude /static_cast<double>(validPairs);

    features.maxFlowMagnitude = globalMaxMagnitude;

    features.upperHorizontalFlow = totalUpperHorizontal / static_cast<double>(validPairs);

    features.upperVerticalFlow = totalUpperVertical / static_cast<double>(validPairs);

    features.flowConvergence = totalConvergence / static_cast<double>(validPairs);

    features.lowerFlowMagnitude = totalLowerMagnitude / static_cast<double>(validPairs);

    features.strideFrequency =  computeStrideFrequency(lowerMotionSignal);

    // check
    std::cout
    << "[OpticalFlow] "
    << sequence.name
    << " | mean=" << features.meanFlowMagnitude
    << " | max=" << features.maxFlowMagnitude
    << " | upperH=" << features.upperHorizontalFlow
    << " | upperV=" << features.upperVerticalFlow
    << " | convergence=" << features.flowConvergence
    << " | lower=" << features.lowerFlowMagnitude
    << std::endl;

    return features;
}