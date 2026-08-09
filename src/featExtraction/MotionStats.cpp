/*
Matteo Gidoni
*/

#include "featExtraction/MotionStats.hpp"

//------------------------------------------------------------

static double motionDensity(
    const cv::Mat& mask,
    const cv::Rect& roi)
{
    if (roi.area() <= 0)
        return 0.0;

    return static_cast<double>(
               cv::countNonZero(mask(roi))
           ) /
           static_cast<double>(roi.area());
}



// ------------------------------------------------------------
// divide l'immagine in tre fasce, alta, media bassa.
// Calcola il numero di pixel bianchi per ogni fascia.
// Puo aiutare a distinguere il tipo di movimento, es. handwaving ha movimento in alto

 MotionStats computeBodyMotion(
    const std::vector<cv::Mat>& motionMasks,
    const std::vector<cv::Rect>& boxes)
{
    MotionStats stats;

    if (motionMasks.size() != boxes.size())
    {
        std::cerr
            << "[computeBodyMotion] SIZE MISMATCH: masks="
            << motionMasks.size()
            << " boxes="
            << boxes.size()
            << std::endl;

        return stats;
    }


    int validFrames = 0;

    double totalLeftPixels = 0.0;
    double totalCenterPixels = 0.0;
    double totalRightPixels = 0.0;

    for (size_t i = 0; i < boxes.size(); i++)
    {
        const cv::Mat& mask = motionMasks[i];
        const cv::Rect& box = boxes[i];

        if (mask.empty() || box.area() <= 0)
            continue;

        cv::Rect imageRect(
            0,
            0,
            mask.cols,
            mask.rows
        );

        // da rivedere questa intersezione
        cv::Rect safeBox = box;

        if (safeBox.width < 3 ||
            safeBox.height < 3)
        {
            continue;
        }

        // ----------------------------
        // Movimento totale nella bbox
        // ----------------------------

        stats.motionEnergy += motionDensity(mask, safeBox);

        // ----------------------------
        // Upper / Middle / Lower
        // ----------------------------

        int h1 = safeBox.height / 3;
        int h2 = safeBox.height / 3;

        cv::Rect upper(
            safeBox.x,
            safeBox.y,
            safeBox.width,
            h1
        );

        cv::Rect middle(
            safeBox.x,
            safeBox.y + h1,
            safeBox.width,
            h2
        );

        cv::Rect lower(
            safeBox.x,
            safeBox.y + h1 + h2,
            safeBox.width,
            safeBox.height - h1 - h2
        );

        stats.upper += motionDensity(mask, upper);
        stats.middle += motionDensity(mask, middle);
        stats.lower += motionDensity(mask, lower);

        // ----------------------------
        // Left / Center / Right
        // ----------------------------

        int w1 = safeBox.width / 3;
        int w2 = safeBox.width / 3;

        cv::Rect left(
            safeBox.x,
            safeBox.y,
            w1,
            safeBox.height
        );

        cv::Rect center(
            safeBox.x + w1,
            safeBox.y,
            w2,
            safeBox.height
        );

        cv::Rect right(
            safeBox.x + w1 + w2,
            safeBox.y,
            safeBox.width - w1 - w2,
            safeBox.height
        );

        stats.left += motionDensity(mask, left);
        stats.right += motionDensity(mask, right);

        totalLeftPixels += cv::countNonZero(mask(left));
        totalCenterPixels += cv::countNonZero(mask(center));
        totalRightPixels += cv::countNonZero(mask(right));

        validFrames++;
    }

    if (validFrames == 0)
        return stats;

    stats.motionEnergy /= validFrames;

    stats.upper /= validFrames;
    stats.middle /= validFrames;
    stats.lower /= validFrames;

    stats.left /= validFrames;
    stats.right /= validFrames;

    double totalHorizontalMotion =
        totalLeftPixels +
        totalCenterPixels +
        totalRightPixels;

    if (totalHorizontalMotion > 0.0)
    {
        stats.centerRatio =
            totalCenterPixels /
            totalHorizontalMotion;
    }

    return stats;
}


