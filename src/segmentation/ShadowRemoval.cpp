#include "segmentation/ShadowRemoval.hpp"

//--------------------------------------------------------------------------------------
// controlla che la distanza verticale della label sopra la testa  sia sotto una percentuale
bool isTooFarAboveHead(
    const cv::Rect& componentBox,
    const cv::Rect& bodyBox,
    double thresholdRatio = 0.20)
{
    if (bodyBox.height <= 0)
        return false;

    int distanceAboveHead =
        bodyBox.y - (componentBox.y + componentBox.height);

    bool isAboveHead = distanceAboveHead > 0;

    if (!isAboveHead)
        return false;

    int maxAllowedAboveHead =
        static_cast<int>(bodyBox.height * thresholdRatio);

    return distanceAboveHead > maxAllowedAboveHead;
}

//--------------------------------------------------------------------------------------
// controlla che la distanza laterale della label sia sotto una percentuale

bool isTooFarLaterally(
    const cv::Rect& componentBox,
    const cv::Rect& bodyBox,
    double thresholdRatio = 0.20)
{
    if (bodyBox.width <= 0)
        return false;

    int lateralDistance = 0;

    // componente completamente a sinistra del corpo
    if (componentBox.x + componentBox.width < bodyBox.x)
    {
        lateralDistance =
            bodyBox.x - (componentBox.x + componentBox.width);
    }
    // componente completamente a destra del corpo
    else if (componentBox.x > bodyBox.x + bodyBox.width)
    {
        lateralDistance =
            componentBox.x - (bodyBox.x + bodyBox.width);
    }
    // componente sovrapposta orizzontalmente al corpo
    else
    {
        lateralDistance = 0;
    }

    double lateralRatio =
        static_cast<double>(lateralDistance) /
        static_cast<double>(bodyBox.width);

    return lateralRatio > thresholdRatio;
}

//--------------------------------------------------------------------------------------


cv::Mat removeShadow(const cv::Mat& mask)
{
    if (mask.empty() || cv::countNonZero(mask) == 0)
        return mask.clone();

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;

    int numLabels = cv::connectedComponentsWithStats(
        mask,
        labels,
        stats,
        centroids,
        8,
        CV_32S
    );

    if (numLabels <= 1)
        return cv::Mat::zeros(mask.size(), CV_8UC1);


    int bodyLabel = -1;
    int bestBodyArea = 0;

    // Cerco il corpo principale, componente più alta che larga
    for (int label = 1; label < numLabels; label++)
    {
        int w = stats.at<int>(label, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(label, cv::CC_STAT_AREA);

        // if (area < 20)
        //     continue;

        bool isVerticalBody = h > w;

        if (isVerticalBody && area > bestBodyArea)
        {
            bestBodyArea = area;
            bodyLabel = label;
        }
    }

    // Se non trovo nessuna componente più alta che larga,
    // considero la mask come ombra/rumore e la elimino
    if (bodyLabel == -1)
        return cv::Mat::zeros(mask.size(), CV_8UC1);

    cv::Rect bodyBox(
        stats.at<int>(bodyLabel, cv::CC_STAT_LEFT),
        stats.at<int>(bodyLabel, cv::CC_STAT_TOP),
        stats.at<int>(bodyLabel, cv::CC_STAT_WIDTH),
        stats.at<int>(bodyLabel, cv::CC_STAT_HEIGHT)
    );

    cv::Mat cleaned = cv::Mat::zeros(mask.size(), CV_8UC1);

    

    for (int label = 1; label < numLabels; label++)
    {
        int x = stats.at<int>(label, cv::CC_STAT_LEFT);
        int y = stats.at<int>(label, cv::CC_STAT_TOP);
        int w = stats.at<int>(label, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        
        // // controllo sulla dimensione dell'area della label
        // if (area > 1)
        //     continue;

        cv::Rect box(x, y, w, h);

        bool isBody = label == bodyLabel;
        bool isBelowHead = box.y >= bodyBox.y;
        bool isHorizontal = w > h;

        // controllo se punti troppo laterali in percentuale rispetto alla larghezza del corpo
        bool tooFarLaterally = isHorizontal && isTooFarLaterally(box, bodyBox, 0.20);

        double widthRatio =
            static_cast<double>(box.width) / static_cast<double>(bodyBox.width);

        // l'ombra deve essere larga almeni il 30% del corpo per essere considerata tale
        bool isLargeEnough = widthRatio > 0.40;

        bool isShadowLike = 
            //isHorizontal  &&
            isBelowHead &&
            isLargeEnough;

        // controllo se punti troppo sopra la testa in percentuale rispetto all'altezza del corpo

        bool tooFarAboveHead =  isTooFarAboveHead(box, bodyBox, 0.20);

        if (isBody)
        {
            cleaned.setTo(255, labels == label);
            continue;
        }

        if (isShadowLike)
            continue;

        if (tooFarAboveHead)
            continue;

        if (tooFarLaterally)
            continue;

        cleaned.setTo(255, labels == label);
    }

    return cleaned;
}

//--------------------------------------------------------------------------------------
