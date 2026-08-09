/*
Matteo Gidoni
*/

#include "segmentation/BoundingBox.hpp"


//--------------------------------------------------------------------------------------
//TROVA LA BBOX perogni MASCHERA FINALE. Prende i diversi contorni e li unisce in una unica BBox

std::vector<cv::Rect> findBBox(
    const std::vector<cv::Mat>& masks,
    int minContourArea )
{
    std::vector<cv::Rect> bboxes;

    for (const auto& mask : masks)
    {
        if (mask.empty())
        {
            bboxes.push_back(cv::Rect());
            continue;
        }

        cv::Mat binary;

        // cv::threshold(mask, binary, 127, 255, cv::THRESH_BINARY);

        // trova i contorni della finalMask
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(
            mask,
            contours,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE
        );

        if (contours.empty())
        {
            bboxes.push_back(cv::Rect());
            continue;
        }

        // controllo se contorni sono validi, cioe area > 20
        // per ora  ho quasi disattivato questo controllo perchè riprosto in removeShadow()
        // ho messo controllo area >1
        cv::Rect finalBox;
        bool hasValidContour = false;

        for (const auto& contour : contours)
        {
            double area = cv::contourArea(contour);

            if (area < minContourArea)
                continue;

            cv::Rect box = cv::boundingRect(contour);

            if (!hasValidContour)
            {
                finalBox = box;
                hasValidContour = true;
            }
            else
            {
                finalBox = finalBox | box;
            }
        }

        if (!hasValidContour)
        {
            bboxes.push_back(cv::Rect());
            continue;
        }

        bboxes.push_back(finalBox);
    }

    return bboxes;
}

//---------------------------------------------------------------------------------------
