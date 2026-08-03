#include "segmentation/GrabCutSegmentation.hpp"
#include "segmentation/ShadowRemoval.hpp"
#include "segmentation/MotionMask.hpp"
#include "segmentation/MediaPipeFallback.hpp"

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

//CREO ROI INIZIALE CENTRALE (SOLO CASO DEL MOVIMENTO MANI)
// Roi grande quasi come l'immagine

cv::Rect getCentralROI(const cv::Mat& img)
{
    int roiWidth  = static_cast<int>(img.cols * 0.8);
    int roiHeight = static_cast<int>(img.rows * 0.9);

    int x = (img.cols - roiWidth) / 2;
    int y = (img.rows - roiHeight) / 2;

    return cv::Rect(x, y, roiWidth, roiHeight);
}

//---------------------------------------------------------------------------------------

//  GRABCUT DA USARE SOLO NEL CASO SIA RICHIAMATO MEDIA PIPE (E' RIDONDANTE)

cv::Mat GrabCutMediaPipe(const cv::Mat& frame, const cv::Rect& roi)
{
    cv::Mat imgForGrabCut;

    if (frame.channels() == 1)
        cv::cvtColor(frame, imgForGrabCut, cv::COLOR_GRAY2BGR);
    else
        imgForGrabCut = frame.clone();

    cv::Rect imgRect(0, 0, imgForGrabCut.cols, imgForGrabCut.rows);
    cv::Rect safeRoi = roi & imgRect;

    if (safeRoi.area() < 50)
        return cv::Mat::zeros(frame.size(), CV_8UC1);

    cv::Mat grabcutMask(imgForGrabCut.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));

    cv::Mat bgModel;
    cv::Mat fgModel;

    cv::grabCut(
        imgForGrabCut,
        grabcutMask,
        safeRoi,
        bgModel,
        fgModel,
        5,
        cv::GC_INIT_WITH_RECT
    );

    cv::Mat finalMask =
        (grabcutMask == cv::GC_FGD) |
        (grabcutMask == cv::GC_PR_FGD);

    finalMask.convertTo(finalMask, CV_8UC1, 255);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(finalMask, finalMask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(finalMask, finalMask, cv::MORPH_CLOSE, kernel);

    return finalMask;
}


//---------------------------------------------------------------------------------------

// GRABCUT CON MOTION MASK
//APPLICO GRABCUT ALLA MOTION MASK SEED ED AL ROI CENTRALE (SOLO PER MOVIM. MANI)
//(MOG2 + MORFOLOGIA BASTANO PER CREARE LA MASCHERA DI RUNNING/ JOGGING/ WALKING)

cv::Mat GrabCut(const cv::Mat& frame, const cv::Rect& roi, const cv::Mat& motionMask)
{
    cv::Mat imgForGrabCut;

    if (frame.channels() == 1)
        cv::cvtColor(frame, imgForGrabCut, cv::COLOR_GRAY2BGR);
    else
        imgForGrabCut = frame.clone();

    cv::Rect imgRect(0, 0, imgForGrabCut.cols, imgForGrabCut.rows);
    cv::Rect safeRoi = roi & imgRect;

    if (safeRoi.area() < 50)
        return cv::Mat::zeros(frame.size(), CV_8UC1);

    if (motionMask.empty() || cv::countNonZero(motionMask) == 0)
        return cv::Mat::zeros(frame.size(), CV_8UC1);

    cv::Mat grabcutMask(imgForGrabCut.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));

    // fuori ROI = background sicuro
    // dentro ROI = probabile persona
    grabcutMask(safeRoi).setTo(cv::GC_PR_FGD);

    cv::Mat cleanSeed;
    motionMask.copyTo(cleanSeed);

    cv::Mat seedInRoi = cv::Mat::zeros(imgForGrabCut.size(), CV_8UC1);
    cleanSeed(safeRoi).copyTo(seedInRoi(safeRoi));

    if (cv::countNonZero(seedInRoi) == 0)
        return cv::Mat::zeros(frame.size(), CV_8UC1);

    // pugni/braccia mossi = foreground sicuro
    grabcutMask.setTo(cv::GC_FGD, seedInRoi > 0);

    int fgCount = cv::countNonZero(grabcutMask == cv::GC_FGD);
    int bgCount = cv::countNonZero(grabcutMask == cv::GC_BGD);

    if (fgCount == 0 || bgCount == 0)
        return cv::Mat::zeros(frame.size(), CV_8UC1);

    cv::Mat bgModel;
    cv::Mat fgModel;

    cv::grabCut(
        imgForGrabCut,
        grabcutMask,
        safeRoi,
        bgModel,
        fgModel,
        5,
        cv::GC_INIT_WITH_MASK
    );

    cv::Mat finalMask =
        (grabcutMask == cv::GC_FGD) |
        (grabcutMask == cv::GC_PR_FGD);

    finalMask.convertTo(finalMask, CV_8UC1, 255);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

    cv::morphologyEx(finalMask, finalMask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(finalMask, finalMask, cv::MORPH_OPEN, kernel);

    return finalMask;
}




//---------------------------------------------------------------------------------------
// da rivedere:

cv::Mat GrabCutModifiedOnlyForShadow(const cv::Mat& frame, const cv::Rect& roi, const cv::Mat& motionMask)
{
    cv::Mat imgForGrabCut;

    if (frame.channels() == 1)
        cv::cvtColor(frame, imgForGrabCut, cv::COLOR_GRAY2BGR);
    else
        imgForGrabCut = frame.clone();

    cv::Rect imgRect(0, 0, imgForGrabCut.cols, imgForGrabCut.rows);
    cv::Rect safeRoi = roi & imgRect;

    if (safeRoi.area() < 50)
        return cv::Mat::zeros(frame.size(), CV_8UC1);

    if (motionMask.empty() || cv::countNonZero(motionMask) == 0)
        return cv::Mat::zeros(frame.size(), CV_8UC1);

    cv::Mat foregroundSeed = createCleanForegroundSeedModified(frame, motionMask);

    if (foregroundSeed.empty() || cv::countNonZero(foregroundSeed) == 0)
        return cv::Mat::zeros(frame.size(), CV_8UC1);

    cv::Mat grabcutMask(imgForGrabCut.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));

    grabcutMask(safeRoi).setTo(cv::GC_PR_BGD);

    cv::Mat darkMask = createDarkMask(frame);

    cv::Mat darkInRoi = cv::Mat::zeros(imgForGrabCut.size(), CV_8UC1);
    darkMask(safeRoi).copyTo(darkInRoi(safeRoi));

    grabcutMask.setTo(cv::GC_PR_FGD, darkInRoi > 0);

    cv::Mat seedInRoi = cv::Mat::zeros(imgForGrabCut.size(), CV_8UC1);
    foregroundSeed(safeRoi).copyTo(seedInRoi(safeRoi));

    if (cv::countNonZero(seedInRoi) == 0)
        return cv::Mat::zeros(frame.size(), CV_8UC1);

    grabcutMask.setTo(cv::GC_FGD, seedInRoi > 0);

    int fgCount = cv::countNonZero(grabcutMask == cv::GC_FGD);
    int bgCount = cv::countNonZero(grabcutMask == cv::GC_BGD);

    if (fgCount == 0 || bgCount == 0)
        return cv::Mat::zeros(frame.size(), CV_8UC1);

    cv::Mat bgModel;
    cv::Mat fgModel;

    cv::grabCut(
        imgForGrabCut,
        grabcutMask,
        safeRoi,
        bgModel,
        fgModel,
        5,
        cv::GC_INIT_WITH_MASK
    );

    cv::Mat finalMask =
        (grabcutMask == cv::GC_FGD) |
        (grabcutMask == cv::GC_PR_FGD);

    finalMask.convertTo(finalMask, CV_8UC1, 255);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

    cv::morphologyEx(finalMask, finalMask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(finalMask, finalMask, cv::MORPH_OPEN, kernel);

    return finalMask;
}

//---------------------------------------------------------------------------------------

// RICAVA LA MASK FINALE (USA GRABCUT O MEDIAPIPE)
// CREO UN FILE DI LOG PER VEDERE QUALI FRAME DANNO PROBLEMI CON LE MOTION MASK
// spostato

std::vector<cv::Mat> finalMask(
    const std::vector<cv::Mat>& frames,
    const std::vector<cv::Mat>& seeds,
    const std::string& sequenceName,
    const bool useMediaPipeForSequence)
{
    std::vector<cv::Mat> result;

    std::ofstream logFile("motion_mask_stats.txt", std::ios::app);

    int mediaPipeCount = 0;
    int emptyMaskCount = 0;


    if (frames.size() != seeds.size())
    {
        std::cerr << "Error: frames and seeds must have the same size." << std::endl;
        return result;
    }

    logFile << "===== NEW SEQUENCE: " << sequenceName << " =====" << std::endl;

    for (size_t i = 0; i < frames.size(); ++i)
    {
        const cv::Mat& frame = frames[i];
        const cv::Mat& motionMask = seeds[i];

        if (frame.empty())
        {
            result.push_back(cv::Mat());
            continue;
        }

        int nonZeroBefore = motionMask.empty() ? 0 : cv::countNonZero(motionMask);

        logFile << " | Frame: " << i
                << " | nonZero BEFORE GrabCut: "
                << nonZeroBefore
                << std::endl;

        cv::Rect roi = getCentralROI(frame);

        cv::Mat mask = GrabCut(frame, roi, motionMask);

        int nonZeroAfter = mask.empty() ? 0 : cv::countNonZero(mask);

        logFile << " | Frame: " << i
                << " | nonZero AFTER GrabCut: "
                << nonZeroAfter
                << std::endl;

        if ((mask.empty() || nonZeroAfter == 0) && useMediaPipeForSequence)
        {
            // emptyMaskCount++;

            // logFile 
            //         << " | Frame: " << i
            //         << " | GrabCut produced EMPTY mask"
            //         << std::endl;

            //     mediaPipeCount++;

            //     logFile << "Sequence: " << sequenceName
            //             << " | Frame: " << i
            //             << " | MediaPipe fallback USED"
            //             << std::endl;

           cv::Rect mediaPipeROI = getMediaPipeROI(frame);

            if (mediaPipeROI.area() > 0)
            {
                mediaPipeCount++;

                logFile 
                        << " | Frame: " << i
                        << " | MediaPipe fallback USED"
                        << std::endl;

                mask = GrabCutMediaPipe(frame, mediaPipeROI);

                int nonZeroAfterMediaPipe =
                    mask.empty() ? 0 : cv::countNonZero(mask);

                logFile 
                        << " | Frame: " << i
                        << " | nonZero AFTER MediaPipe GrabCut: "
                        << nonZeroAfterMediaPipe
                        << std::endl;
            }
            else
            {
                logFile 
                        << " | Frame: " << i
                        << " | MediaPipe FAILED"
                        << std::endl;
            }
        }

        result.push_back(mask);

        logFile << "----------------------------------------" << std::endl;
    }

    logFile << "=========== SUMMARY ===========" << std::endl;
    logFile << "Sequence: " << sequenceName << std::endl;
    logFile << "Empty masks after GrabCut: " << emptyMaskCount << std::endl;
    logFile << "MediaPipe fallback used: " << mediaPipeCount << std::endl;
    logFile << "================================" << std::endl << std::endl;

    logFile.close();

    return result;
}

//---------------------------------------------------------------------------------------
