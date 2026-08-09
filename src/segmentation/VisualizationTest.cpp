/*
Matteo Gidoni
*/

#include "segmentation/VisualizationTest.hpp"

//--------------------------------------------------------------------------------------

// MOSTRA BBOX DISEGNATE

void showBBoxes(
    const std::vector<cv::Mat>& frames,
    const std::vector<cv::Rect>& bboxes)
{
    for (size_t i = 0; i < frames.size() && i < bboxes.size(); i++)
    {
        if (frames[i].empty())
            continue;

        cv::Mat display;

        if (frames[i].channels() == 1)
            cv::cvtColor(frames[i], display, cv::COLOR_GRAY2BGR);
        else
            display = frames[i].clone();

        if (bboxes[i].area() > 0)
        {
            cv::rectangle(
                display,
                bboxes[i],
                cv::Scalar(0, 255, 0),
                2
            );
        }

        // salva il rettangolo in sequence
        cv::imshow("Bounding Boxes", display);

        int key = cv::waitKey(30);

        if (key == 27)
            break;
    }

    cv::destroyAllWindows();
}

//--------------------------------------------------------------------------------------
// VISUALIZZA MASCHERA FINALE

void showMasks(const std::vector<cv::Mat>& masks)
{
    for (size_t i = 0; i < masks.size(); i++)
    {
        cv::imshow("MOG2 Mask", masks[i]);

        // attende 30 ms tra un frame e l'altro
        int key = cv::waitKey(30);

        // ESC per uscire
        if (key == 27)
            break;
    }

    cv::destroyAllWindows();
}

//---------------------------------------------------------------------------------------
