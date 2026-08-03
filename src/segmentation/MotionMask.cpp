#include "segmentation/MotionMask.hpp"
#include "segmentation/ShadowRemoval.hpp"
#include <fstream>
#include <cstdlib>
#include <cmath>
//--------------------------------------------------------------------------------------

// FILL TEMPORALE: RIEMPIO LE MOTION MASK NERE CON MOTION MASK BUONE NELLA STESSA SEQUENZA

std::vector<cv::Mat> fillEmptySeedsTemporal(
    const std::vector<cv::Mat>& seeds,
    int minNonZero)
{
    std::vector<cv::Mat> filled = seeds;

    if (seeds.empty())
        return filled;

    int n = static_cast<int>(seeds.size());

    std::vector<int> validIndices;

    for (int i = 0; i < n; i++)
    {
        int nonZero = seeds[i].empty() ? 0 : cv::countNonZero(seeds[i]);

        if (nonZero > minNonZero)
            validIndices.push_back(i);
    }

    // Se tutta la sequenza è vuota, non posso ricostruire temporalmente
    if (validIndices.empty())
        return filled;

    for (int i = 0; i < n; i++)
    {
        int nonZero = seeds[i].empty() ? 0 : cv::countNonZero(seeds[i]);

        if (nonZero > minNonZero)
            continue;

        int nearestIndex = validIndices[0];
        int bestDistance = std::abs(i - nearestIndex);

        for (int idx : validIndices)
        {
            int distance = std::abs(i - idx);

            if (distance < bestDistance)
            {
                bestDistance = distance;
                nearestIndex = idx;
            }
        }

        filled[i] = seeds[nearestIndex].clone();
    }

    return filled;
}


//---------------------------------------------------------------------------------------
// da rivedere: 

cv::Mat createCleanForegroundSeedModified(
    const cv::Mat& frame,
    const cv::Mat& motionMask)
{
    if (frame.empty() || motionMask.empty())
        return cv::Mat();

    cv::Mat cleanMotion;

    if (motionMask.channels() == 3)
        cv::cvtColor(motionMask, cleanMotion, cv::COLOR_BGR2GRAY);
    else
        cleanMotion = motionMask.clone();

    cv::threshold(cleanMotion, cleanMotion, 200, 255, cv::THRESH_BINARY);

    cv::Mat darkMask = createDarkMask(frame);

    cv::Mat foregroundSeed;
    cv::bitwise_and(cleanMotion, darkMask, foregroundSeed);

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(3, 3)
    );

    cv::morphologyEx(foregroundSeed, foregroundSeed, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(foregroundSeed, foregroundSeed, cv::MORPH_DILATE, kernel);

    return foregroundSeed;
}

//---------------------------------------------------------------------------------------
// filtro con morfologia

std::vector<cv::Mat> filterMorphology(const std::vector<cv::Mat>& frames)
{
    std::vector<cv::Mat> result;

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));

    for (const auto& frame : frames){
        cv::Mat temp;

        cv::threshold(frame, temp, 200, 255, cv::THRESH_BINARY);
    // sarebbe meglio fare close e poi open?
    // Rimuove piccoli rumori bianchi
        cv::morphologyEx(
            temp,
            temp,
            cv::MORPH_OPEN,
            kernel
        );

        // Chiude piccoli buchi negli oggetti
        cv::morphologyEx(
            temp,
            temp,
            cv::MORPH_CLOSE,
            kernel
        );

        result.push_back(temp);
    }
 
    return result;
}

//---------------------------------------------------------------------------------------

//FUNZIONE PER TOGLIERE CAMBI DI LUCE

bool isLightChangeMask(const cv::Mat& mask, double threshold)
{
    if (mask.empty())
        return false;

    int nonZero = cv::countNonZero(mask);

    double ratio =
        static_cast<double>(nonZero) /
        static_cast<double>(mask.total());

    return ratio > threshold;
}

//---------------------------------------------------------------------------------------

// creo filtro MOG2: separa il foreground da background. Ritorno il foreground come Mat

 std::vector<cv::Mat> filterMOG2(const std::vector<cv::Mat>& frames)
{
    std::vector<cv::Mat> foreground;

    std::ofstream logFile("motion_mask_stats.txt", std::ios::app);

    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2 =
        cv::createBackgroundSubtractorMOG2(
            500,
            16.0,
            true
        );

    cv::Mat lastGoodMask;
    bool hasLastGoodMask = false;

    for (size_t i = 0; i < frames.size(); i++)
    {
        const cv::Mat& frame = frames[i];

        if (frame.empty())
        {
            foreground.push_back(cv::Mat());
            continue;
        }

        cv::Mat rawMask;
        mog2->apply(frame, rawMask);

        cv::Mat mask;

        // Tengo solo foreground vero, escludendo shadow = 127
        cv::inRange(rawMask, 254, 255, mask);

        int nonZero = cv::countNonZero(mask);

        double ratio =
            static_cast<double>(nonZero) /
            static_cast<double>(mask.total());

        logFile << "[MOG2] Frame: " << i
                << " | nonZero: " << nonZero
                << " | ratio: " << ratio
                << std::endl;

        bool lightChange = isLightChangeMask(mask, 0.35);

        if (lightChange)
        {
            logFile << "[MOG2] Frame: " << i
                    << " | LIGHT CHANGE DETECTED"
                    << std::endl;

            if (hasLastGoodMask)
            {
                foreground.push_back(lastGoodMask.clone());
            }
            else
            {
                foreground.push_back(cv::Mat::zeros(mask.size(), CV_8UC1));
            }

            continue;
        }

        if (nonZero > 0)
        {
            lastGoodMask = mask.clone();
            hasLastGoodMask = true;
        }

        foreground.push_back(mask);
    }

    logFile << "----------------------------------------" << std::endl;
    logFile.close();

    return foreground;
}

//---------------------------------------------------------------------------------------

