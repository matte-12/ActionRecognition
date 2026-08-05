#include "segmentation/ShadowRemoval.hpp"



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

        double widthRatio =
            static_cast<double>(box.width) / static_cast<double>(bodyBox.width);

        // l'ombra deve essere larga almeni il 30% del corpo per essere considerata tale
        bool isLargeEnough = widthRatio > 0.50;

        bool isShadowLike = isHorizontal && isBelowHead && isLargeEnough;

        // controllo se punti troppo sopra la testa

        // distanza verticale tra la parte bassa della componente
        // e la parte alta del corpo
        int distanceAboveHead =
            bodyBox.y - (box.y + box.height);

        // se distanceAboveHead > 0, la componente è sopra la testa
        bool isAboveHead = distanceAboveHead > 0;

        // soglia proporzionale all'altezza del corpo
        double maxAboveHeadRatio = 0.20;

        int maxAllowedAboveHead =
            static_cast<int>(bodyBox.height * maxAboveHeadRatio);

        bool tooFarAboveHead =
            isAboveHead &&
            distanceAboveHead > maxAllowedAboveHead;

        if (isBody)
        {
            cleaned.setTo(255, labels == label);
            continue;
        }

        if (isShadowLike)
            continue;

        if (tooFarAboveHead)
            continue;

        cleaned.setTo(255, labels == label);
    }

    return cleaned;
}

