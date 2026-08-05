#include "segmentation/GrabCutSegmentation.hpp"
#include "segmentation/ShadowRemoval.hpp"
#include "segmentation/MotionMask.hpp"
#include "segmentation/MediaPipeFallback.hpp"

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

//CREO ROI INIZIALE CENTRALE (SOLO CASO DEL MOVIMENTO MANI, altrimenti crea mask nere)
// Roi grande quasi come l'immagine

cv::Rect getCentralROI(const cv::Mat& img)
{
    int roiWidth  = static_cast<int>(img.cols * 0.9);
    int roiHeight = static_cast<int>(img.rows * 0.99);

    int x = (img.cols - roiWidth) / 2;
    int y = (img.rows - roiHeight) / 2;

    return cv::Rect(x, y, roiWidth, roiHeight);
}

//---------------------------------------------------------------------------------------

// determino se assegnare GC_PR_BGD o GC_PR_FGD alla safeRoi

bool useBGDseed(const cv::Mat& seedInRoi, const cv::Rect& roi)
{
    int nonZero = cv::countNonZero(seedInRoi);

    double ratio =
        static_cast<double>(nonZero) / static_cast<double>(roi.area());

    bool conservative = ratio < 0.08;

    std::ofstream logFile("motion_mask_stats.txt", std::ios::app);
    logFile << "[GC seed] seedRatioInRoi: "
          << ratio
          << " | mode: "
          << (conservative ? "PR_FGD": "PR_BGD" )
          << std::endl;
    logFile.close();
    return conservative;
}

//---------------------------------------------------------------------------------------

// determino se assegnare GC_PR_BGD o GC_PR_FGD alla finalMask

bool useBGDfinalMask(const cv::Mat& finalMask, const cv::Rect& roi)
{
    int nonZero = cv::countNonZero(finalMask(roi));

    double finalRatio =
        static_cast<double>(nonZero) / static_cast<double>(roi.area());

    bool redo = finalRatio > 0.18;

    std::ofstream logFile("motion_mask_stats.txt", std::ios::app);
    logFile << "[GC finalMask] final ratio: "
          << finalRatio
          << " | mode: "
          << (redo ? "PR_BGD" : "PR_FGD")
          << std::endl;
    logFile.close();
    return redo;
}

//---------------------------------------------------------------------------------------

// trovo la final mask. Esegue il vero GrabCut

cv::Mat runGrabCut(
    const cv::Mat& img,
    const cv::Rect& roi,
    const cv::Mat& seedInRoi,
    bool flag)
{
    cv::Mat grabcutMask(img.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));

    if (flag)
        grabcutMask(roi).setTo(cv::GC_PR_FGD);
    else
        grabcutMask(roi).setTo(cv::GC_PR_BGD);

    grabcutMask.setTo(cv::GC_FGD, seedInRoi > 0);

    cv::Mat bgModel, fgModel;

    cv::grabCut(
        img,
        grabcutMask,
        roi,
        bgModel,
        fgModel,
        5,
        cv::GC_INIT_WITH_MASK
    );

    cv::Mat finalMask =
        (grabcutMask == cv::GC_FGD) |
        (grabcutMask == cv::GC_PR_FGD);

    finalMask.convertTo(finalMask, CV_8UC1, 255);

    return finalMask;
}

//---------------------------------------------------------------------------------------

// grabcut mediapipe

cv::Mat GrabCutMediaPipe(
    const cv::Mat& frame,
    const cv::Rect& roi)
{
    cv::Mat img;

    if (frame.channels() == 1)
        cv::cvtColor(frame, img, cv::COLOR_GRAY2BGR);
    else
        img = frame.clone();

    if (img.empty())
        return cv::Mat();

    cv::Rect imageRect(0, 0, img.cols, img.rows);
    cv::Rect safeRoi = roi & imageRect;

    if (safeRoi.area() < 50)
        return cv::Mat::zeros(frame.size(), CV_8UC1);

    cv::Mat grabcutMask(img.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));

    cv::Mat bgModel;
    cv::Mat fgModel;

    cv::grabCut(
        img,
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

    cv::Mat kernel =
        cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(3, 3)
        );

    cv::morphologyEx(finalMask, finalMask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(finalMask, finalMask, cv::MORPH_OPEN, kernel);

    return finalMask;
}

//---------------------------------------------------------------------------------------

// GRABCUT CON MOTION MASK
//APPLICO GRABCUT ALLA MOTION MASK SEED ED AL ROI CENTRALE (SOLO PER MOVIM. MANI)
//(MOG2 + MORFOLOGIA BASTANO PER CREARE LA MASCHERA DI RUNNING/ JOGGING/ WALKING)

cv::Mat GrabCut(const cv::Mat& frame, const cv::Rect& roi, const cv::Mat& motionMask)
{
    cv::Mat img;

    if (frame.channels() == 1)
        cv::cvtColor(frame, img, cv::COLOR_GRAY2BGR);
    else
        img = frame.clone();

     cv::Rect imgRect(0, 0, img.cols, img.rows);
     cv::Rect safeRoi = roi & imgRect;

     if (safeRoi.area() < 50)
        return cv::Mat::zeros(frame.size(), CV_8UC1);

    if (motionMask.empty() || cv::countNonZero(motionMask) == 0)
        return cv::Mat::zeros(frame.size(), CV_8UC1);

    cv::Mat grabcutMask(img.size(), CV_8UC1, cv::Scalar(cv::GC_BGD));



    cv::Mat seedInRoi = cv::Mat::zeros(img.size(), CV_8UC1);
    motionMask(safeRoi).copyTo(seedInRoi(safeRoi));

        // pugni/braccia mossi = foreground sicuro
    grabcutMask.setTo(cv::GC_FGD, seedInRoi > 0);

    if (cv::countNonZero(seedInRoi) == 0)
        return cv::Mat::zeros(frame.size(), CV_8UC1);

    // fuori ROI = background sicuro
    // dentro ROI = probabile persona
    // lo determino in modo dinamico con il ratio

    //bool conservative = useBGDseed(seedInRoi, safeRoi);
    bool conservative = true;

    cv::Mat finalMask =
        runGrabCut(
            img,
            safeRoi,
            seedInRoi,
            conservative
        );
    
    // bool che mi dice se i punti bianchi della final mask sono elevati
     bool redo = useBGDfinalMask(finalMask, safeRoi);
       // bool redo = false;


        
    //
    if (redo && conservative)
    {
        finalMask =
            runGrabCut(
                img,
                safeRoi,
                seedInRoi,
                false
            );
    }


    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

    cv::morphologyEx(finalMask, finalMask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(finalMask, finalMask, cv::MORPH_OPEN, kernel);

    return finalMask;
}


//---------------------------------------------------------------------------------------

// RICAVA LA MASK FINALE (USA GRABCUT O MEDIAPIPE)
// CREO UN FILE DI LOG PER VEDERE QUALI FRAME DANNO PROBLEMI CON LE MOTION MASK
// spostato

std::vector<cv::Mat> getFinalMask(
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
    std::cerr << "frames.size() = " << frames.size() << std::endl;
    std::cerr << "seeds.size()  = " << seeds.size() << std::endl;

    return result;
}

    logFile << "===== SEQUENCE: " << sequenceName << " =====" << std::endl;

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

        // ROI = regione entro l'immagine in cui si presume ci sia qualcuno
        cv::Rect roi = getCentralROI(frame);

        // applico algoritmo GrabCut
        cv::Mat mask = GrabCut(frame, roi, motionMask);

        int nonZeroAfter = mask.empty() ? 0 : cv::countNonZero(mask);

        logFile << " | Frame: " << i
                << " | nonZero AFTER GrabCut: "
                << nonZeroAfter
                << std::endl;

        // se maschere nere allora usa mediapipe
        if ((mask.empty() || nonZeroAfter == 0) && useMediaPipeForSequence)
        {

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
