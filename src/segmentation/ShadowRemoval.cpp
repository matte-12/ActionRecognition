#include "segmentation/ShadowRemoval.hpp"
#include "segmentation/GrabCutSegmentation.hpp"
#include "segmentation/BoundingBox.hpp"

#include <fstream>
#include <cstdlib>
#include <cmath>

// DARK MASK PER ELIMINARE LE OMBRE

cv::Mat createDarkMask(const cv::Mat& frame) {
    cv::Mat gray;

    if (frame.channels() == 3)
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else
        gray = frame.clone();

    cv::Mat darkMask;

    // Valore da tarare: 70/80/90 a seconda del dataset
    cv::threshold(gray, darkMask, 80, 255, cv::THRESH_BINARY_INV);

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(3, 3)
    );

    cv::morphologyEx(darkMask, darkMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(darkMask, darkMask, cv::MORPH_CLOSE, kernel);

    return darkMask;
}


//---------------------------------------------------------------------------------------
// da rivedere:
bool isShadowOrAbruptChangeFrame(
    const cv::Rect& currentBox,
    const cv::Rect& previousGoodBox,
    const cv::Mat& currentMask,
    const cv::Size& frameSize)
{
    if (currentBox.area() <= 0)
        return false;

    double imageArea =
        static_cast<double>(frameSize.width * frameSize.height);

    double areaRatio =
        static_cast<double>(currentBox.area()) / imageArea;

    double aspect =
        static_cast<double>(currentBox.width) /
        static_cast<double>(currentBox.height);

    if (areaRatio > 0.35)
        return true;

    if (aspect > 1.35)
        return true;

    if (!currentMask.empty())
    {
        int nonZero = cv::countNonZero(currentMask);

        double maskRatio =
            static_cast<double>(nonZero) /
            static_cast<double>(currentMask.total());

        if (maskRatio > 0.35)
            return true;
    }

    if (previousGoodBox.area() > 0)
    {
        double growth =
            static_cast<double>(currentBox.area()) /
            static_cast<double>(previousGoodBox.area());

        if (growth > 2.3)
            return true;

        if (currentBox.width > previousGoodBox.width * 1.8)
            return true;

        cv::Point2d c1(
            currentBox.x + currentBox.width / 2.0,
            currentBox.y + currentBox.height / 2.0
        );

        cv::Point2d c2(
            previousGoodBox.x + previousGoodBox.width / 2.0,
            previousGoodBox.y + previousGoodBox.height / 2.0
        );

        double dx = std::abs(c1.x - c2.x);
        double dy = std::abs(c1.y - c2.y);

        if (dx > previousGoodBox.width * 2.0)
            return true;

        if (dy > previousGoodBox.height * 1.2)
            return true;
    }

    return false;
}

//---------------------------------------------------------------------------------------
// da rivedere:
bool shouldAcceptModifiedShadowResult(
    const cv::Rect& originalBox,
    const cv::Rect& modifiedBox,
    const cv::Rect& previousGoodBox,
    const cv::Size& frameSize)
{
    if (modifiedBox.area() <= 0)
        return false;

    if (originalBox.area() <= 0)
        return false;

    double imageArea =
        static_cast<double>(frameSize.width * frameSize.height);

    double modifiedAreaRatio =
        static_cast<double>(modifiedBox.area()) / imageArea;

    double modifiedAspect =
        static_cast<double>(modifiedBox.width) /
        static_cast<double>(modifiedBox.height);

    if (modifiedAreaRatio < 0.002)
        return false;

    if (modifiedAreaRatio > 0.45)
        return false;

    if (modifiedAspect < 0.08)
        return false;

    if (modifiedAspect > 1.80)
        return false;

    if (modifiedBox.area() >= originalBox.area() * 0.98)
        return false;

    if (modifiedBox.area() < originalBox.area() * 0.05)
        return false;

    if (previousGoodBox.area() > 0)
    {
        if (modifiedBox.height < previousGoodBox.height * 0.50)
            return false;

        cv::Point2d c1(
            modifiedBox.x + modifiedBox.width / 2.0,
            modifiedBox.y + modifiedBox.height / 2.0
        );

        cv::Point2d c2(
            previousGoodBox.x + previousGoodBox.width / 2.0,
            previousGoodBox.y + previousGoodBox.height / 2.0
        );

        double dx = std::abs(c1.x - c2.x);
        double dy = std::abs(c1.y - c2.y);

        if (dx > previousGoodBox.width * 2.2)
            return false;

        if (dy > previousGoodBox.height * 1.3)
            return false;
    }

    return true;
}

//---------------------------------------------------------------------------------------
// da rivedere: funzione per correggere solo i frame sospetti di shadow con GrabCut modificato
std::vector<cv::Mat> correctOnlyShadowFramesWithModifiedSegmentation(
    const std::vector<cv::Mat>& frames,
    const std::vector<cv::Mat>& seeds,
    const std::vector<cv::Mat>& originalMasks,
    std::vector<cv::Rect>& bboxes,
    const std::string& sequenceName)
{
    std::vector<cv::Mat> correctedMasks = originalMasks;

    std::ofstream logFile("shadow_switch_stats.txt", std::ios::app);

    logFile << "===== NEW SEQUENCE: " << sequenceName << " =====" << std::endl;

    cv::Rect previousGoodBox;

    for (size_t i = 0; i < frames.size() && i < seeds.size() && i < correctedMasks.size() && i < bboxes.size(); i++)
    {
        const cv::Mat& frame = frames[i];

        if (frame.empty())
            continue;

        cv::Rect currentBox = bboxes[i];

        bool suspicious =
            isShadowOrAbruptChangeFrame(
                currentBox,
                previousGoodBox,
                correctedMasks[i],
                frame.size()
            );

        logFile << "Frame: " << i
                << " | suspicious: " << suspicious
                << " | originalBox: "
                << currentBox.x << " "
                << currentBox.y << " "
                << currentBox.width << " "
                << currentBox.height
                << std::endl;

        if (suspicious)
        {
            cv::Rect roi = getCentralROI(frame);

            cv::Mat modifiedMask =
                GrabCutModifiedOnlyForShadow(
                    frame,
                    roi,
                    seeds[i]
                );
// da rivedere l'int 20 come parametro minimo per trovare la bbox
            int i = 20;
            cv::Rect modifiedBox =
                findSingleBBoxOriginalLogic(modifiedMask, i);

            bool accept =
                shouldAcceptModifiedShadowResult(
                    currentBox,
                    modifiedBox,
                    previousGoodBox,
                    frame.size()
                );

            logFile << "Frame: " << i
                    << " | modifiedBox: "
                    << modifiedBox.x << " "
                    << modifiedBox.y << " "
                    << modifiedBox.width << " "
                    << modifiedBox.height
                    << " | accept: " << accept
                    << std::endl;

            if (accept)
            {
                correctedMasks[i] = modifiedMask;
                bboxes[i] = modifiedBox;
                currentBox = modifiedBox;
            }
        }

        bool currentStillSuspicious =
            isShadowOrAbruptChangeFrame(
                currentBox,
                previousGoodBox,
                correctedMasks[i],
                frame.size()
            );

        if (currentBox.area() > 0 && !currentStillSuspicious)
            previousGoodBox = currentBox;

        logFile << "----------------------------------------" << std::endl;
    }

    logFile << "================================" << std::endl << std::endl;

    logFile.close();

    return correctedMasks;
}

//---------------------------------------------------------------------------------------
