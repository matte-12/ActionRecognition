#include "segmentation/BoundingBox.hpp"

// // FILTRO: LA DIMENSIONE DELLA BBOX DEVE ESSERE DI GRANDEZZA UMANA
// non è utile perchè la box puo avere qualsiasi dimensione.

// bool isHumanLikeBox(const cv::Rect& box, const cv::Size& frameSize) {
//     if (box.area() <= 0) return false;

//     double imageArea = frameSize.width * frameSize.height;
//     double areaRatio = static_cast<double>(box.area()) / imageArea;

//     double aspect = static_cast<double>(box.width) / box.height;

//     if (areaRatio < 0.002) return false;  // troppo piccolo
//     if (areaRatio > 0.35) return false;   // troppo grande, probabile ombra/sfondo

//     if (aspect < 0.15) return false;      // troppo stretto
//     if (aspect > 0.90) return false;      // troppo largo per una persona verticale

//     return true;
// }

// //--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------

// // FACENDO SMOOTH EVITA CHE CI SIANO CAMBI BRUSCHI NELLE BBOX
// // (NON LA STO USANDO, NON MI HA CONVINTO)

// std::vector<cv::Rect> smoothBBoxes(
//     const std::vector<cv::Rect>& bboxes,
//     double alpha ,
//     int maxEmptyGap )
// {
//     std::vector<cv::Rect> smoothed;

//     if (bboxes.empty())
//         return smoothed;

//     cv::Rect2d lastBox;
//     bool hasLastBox = false;

//     int emptyGap = 0;

//     for (size_t i = 0; i < bboxes.size(); i++)
//     {
//         const cv::Rect& current = bboxes[i];

//         // Caso bbox vuota
//         if (current.area() <= 0)
//         {
//             emptyGap++;

//             // Se mancano pochi frame, mantengo la bbox precedente
//             if (hasLastBox && emptyGap <= maxEmptyGap)
//             {
//                 smoothed.push_back(cv::Rect(
//                     static_cast<int>(lastBox.x),
//                     static_cast<int>(lastBox.y),
//                     static_cast<int>(lastBox.width),
//                     static_cast<int>(lastBox.height)
//                 ));
//             }
//             else
//             {
//                 smoothed.push_back(cv::Rect());
//                 hasLastBox = false;
//             }

//             continue;
//         }

//         emptyGap = 0;

//         cv::Rect2d currentBox(
//             current.x,
//             current.y,
//             current.width,
//             current.height
//         );

//         // Primo box valido
//         if (!hasLastBox)
//         {
//             lastBox = currentBox;
//             hasLastBox = true;

//             smoothed.push_back(current);
//             continue;
//         }

//         // Exponential smoothing
//         cv::Rect2d newBox;

//         newBox.x =
//             alpha * currentBox.x + (1.0 - alpha) * lastBox.x;

//         newBox.y =
//             alpha * currentBox.y + (1.0 - alpha) * lastBox.y;

//         newBox.width =
//             alpha * currentBox.width + (1.0 - alpha) * lastBox.width;

//         newBox.height =
//             alpha * currentBox.height + (1.0 - alpha) * lastBox.height;

//         lastBox = newBox;

//         smoothed.push_back(cv::Rect(
//             static_cast<int>(newBox.x),
//             static_cast<int>(newBox.y),
//             static_cast<int>(newBox.width),
//             static_cast<int>(newBox.height)
//         ));
//     }

//     return smoothed;
// }

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
