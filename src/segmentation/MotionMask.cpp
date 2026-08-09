/*
Matteo Gidoni
*/

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
// filtro con morfologia

std::vector<cv::Mat> filterMorphology(const std::vector<cv::Mat>& frames)
{
    std::vector<cv::Mat> result;

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));

    for (const auto& frame : frames){
 
    // sarebbe meglio fare close e poi open?
    // Rimuove piccoli rumori bianchi
        cv::morphologyEx(
            frame,
            frame,
            cv::MORPH_OPEN,
            kernel
        );

        // Chiude piccoli buchi negli oggetti
        cv::morphologyEx(
            frame,
            frame,
            cv::MORPH_CLOSE,
            kernel
        );

        result.push_back(frame);
    }
 
    return result;
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
        
    // trick per riempire i cambi di luce. Uso l'ultima mask buona.
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
        // ottengo foregroung grezzo 
        cv::Mat rawMask;
        mog2->apply(frame, rawMask);

        cv::Mat mask;

        // Tengo solo foreground vero, escludendo shadow = 127
        cv::inRange(rawMask, 254, 255, mask);

        // conta numero pixel non neri, se il numero è basso allora 
        //o non c'è la persona oppure lo rileva male.In questo caso uso la mask dell'ultimo frame buono, se c'è
        int nonZero = cv::countNonZero(mask);

        double ratio =
            static_cast<double>(nonZero) /
            static_cast<double>(mask.total());

        logFile << "[MOG2] Frame: " << i
                << " | nonZero: " << nonZero
                << " | ratio: " << ratio
                << std::endl;


        foreground.push_back(mask);
    }

    logFile << "----------------------------------------" << std::endl;
    logFile.close();

    return foreground;
}

//---------------------------------------------------------------------------------------

