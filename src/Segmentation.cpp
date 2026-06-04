#include "Segmentation.hpp"
#include <fstream>
#include <cstdlib>

// NB HA ANCORA PROBLEMI CON OMBRE E CAMBI DI LUCE
//--------------------------------------------------------------------------------------

// FACENDO SMOOTH EVITA CHE CI SIANO CAMBI BRUSCHI NELLE BBOX
// (NON LA STO USANDO, NON MI HA CONVINTO)

std::vector<cv::Rect> smoothBBoxes(
    const std::vector<cv::Rect>& bboxes,
    double alpha = 0.35,
    int maxEmptyGap = 3)
{
    std::vector<cv::Rect> smoothed;

    if (bboxes.empty())
        return smoothed;

    cv::Rect2d lastBox;
    bool hasLastBox = false;

    int emptyGap = 0;

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const cv::Rect& current = bboxes[i];

        // Caso bbox vuota
        if (current.area() <= 0)
        {
            emptyGap++;

            // Se mancano pochi frame, mantengo la bbox precedente
            if (hasLastBox && emptyGap <= maxEmptyGap)
            {
                smoothed.push_back(cv::Rect(
                    static_cast<int>(lastBox.x),
                    static_cast<int>(lastBox.y),
                    static_cast<int>(lastBox.width),
                    static_cast<int>(lastBox.height)
                ));
            }
            else
            {
                smoothed.push_back(cv::Rect());
                hasLastBox = false;
            }

            continue;
        }

        emptyGap = 0;

        cv::Rect2d currentBox(
            current.x,
            current.y,
            current.width,
            current.height
        );

        // Primo box valido
        if (!hasLastBox)
        {
            lastBox = currentBox;
            hasLastBox = true;

            smoothed.push_back(current);
            continue;
        }

        // Exponential smoothing
        cv::Rect2d newBox;

        newBox.x =
            alpha * currentBox.x + (1.0 - alpha) * lastBox.x;

        newBox.y =
            alpha * currentBox.y + (1.0 - alpha) * lastBox.y;

        newBox.width =
            alpha * currentBox.width + (1.0 - alpha) * lastBox.width;

        newBox.height =
            alpha * currentBox.height + (1.0 - alpha) * lastBox.height;

        lastBox = newBox;

        smoothed.push_back(cv::Rect(
            static_cast<int>(newBox.x),
            static_cast<int>(newBox.y),
            static_cast<int>(newBox.width),
            static_cast<int>(newBox.height)
        ));
    }

    return smoothed;
}

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

        cv::imshow("Bounding Boxes", display);

        int key = cv::waitKey(30);

        if (key == 27)
            break;
    }

    cv::destroyAllWindows();
}

//--------------------------------------------------------------------------------------

// FUNZIONE PER STABILIRE SE DOVREI USARE MEDIAPIPE O NO, IN BASE ALLA % DI SEEDS BUONI

bool shouldUseMediaPipeForSequence(
    const std::vector<cv::Mat>& seeds,
    double zeroRatioThreshold = 0.80)
{
    if (seeds.empty())
        return true;

    int zeroCount = 0;

    for (const auto& seed : seeds)
    {
        int nonZero = seed.empty() ? 0 : cv::countNonZero(seed);

        if (nonZero == 0)
            zeroCount++;
    }

    double zeroRatio =
        static_cast<double>(zeroCount) /
        static_cast<double>(seeds.size());

    return zeroRatio >= zeroRatioThreshold;
}

//--------------------------------------------------------------------------------------

// FILL TEMPORALE: RIEMPIO LE MOTION MASK NERE CON MOTION MASK BUONE NELLA STESSA SEQUENZA

std::vector<cv::Mat> fillEmptySeedsTemporal(
    const std::vector<cv::Mat>& seeds,
    int minNonZero = 10)
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
//--------------------------------------------------------------------------------------
// USO MEDIAPIPE COME FALLBACK PER IL CASO DI CAMBIO DI SCALA (D2)
// SE MOTION MASK NERA ALLORA USA MEDIAPIPE

cv::Rect getMediaPipeROI(const cv::Mat& frame)
{
    std::string imgPath = "temp_mediapipe_frame.jpg";
    std::string outPath = "temp_mediapipe_roi.txt";

    cv::imwrite(imgPath, frame);

    std::string scriptPath =
        "/home/roaming/gidomat18904/comp_vision/ActionRecognition/mp_roi.py";

    std::string modelPath =
        "/home/roaming/gidomat18904/comp_vision/ActionRecognition/pose_landmarker_lite.task";

    std::string command =
        "python3 " + scriptPath + " " + imgPath + " " + modelPath + " > " + outPath;

    int ret = std::system(command.c_str());

    if (ret != 0)
        return cv::Rect();

    std::ifstream file(outPath);

    int x, y, w, h;
    file >> x >> y >> w >> h;

    if (w <= 0 || h <= 0)
        return cv::Rect();

    cv::Rect roi(x, y, w, h);
    cv::Rect imgRect(0, 0, frame.cols, frame.rows);

    return roi & imgRect;
}


//--------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------

//CREO ROI INIZIALE CENTRALE (SOLO CASO DEL MOVIMENTO MANI)


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
// RICAVA LA MASK FINALE (USA GRABCUT O MEDIAPIPE)
// CREO UN FILE DI LOG PER VEDERE QUALI FRAME DANNO PROBLEMI CON LE MOTION MASK


std::vector<cv::Mat> Segmentation::finalMask(
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

// filtro con morfologia

std::vector<cv::Mat> Segmentation:: filterMorphology(const std::vector<cv::Mat>& frames)
{
    std::vector<cv::Mat> result;

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));

    for (const auto& frame : frames){
        cv::Mat temp;

        cv::threshold(frame, temp, 200, 255, cv::THRESH_BINARY);

    // Rimuove piccoli rumori bianchi
        cv::morphologyEx(
            temp,
            temp,
            cv::MORPH_OPEN,
            kernel
        );

        // Chiude piccoli buchi negli oggetti
        cv::morphologyEx(
            temp,
            temp,
            cv::MORPH_CLOSE,
            kernel
        );

        result.push_back(temp);
    }
 
    return result;
}

//---------------------------------------------------------------------------------------

//FUNZIONE PER TOGLIERE CAMBI DI LUCE

bool isLightChangeMask(const cv::Mat& mask, double threshold = 0.35)
{
    if (mask.empty())
        return false;

    int nonZero = cv::countNonZero(mask);

    double ratio =
        static_cast<double>(nonZero) /
        static_cast<double>(mask.total());

    return ratio > threshold;
}

//---------------------------------------------------------------------------------------

//FUNZIONE PER TOGLIERE LE OMBRE

cv::Mat suppressShadowsByBackground(
    const cv::Mat& frame,
    const cv::Mat& rawMask,
    const cv::Mat& background)
{
    cv::Mat fgMask;
    cv::inRange(rawMask, 254, 255, fgMask); 
    // tengo solo foreground sicuro, escludo 127 di MOG2

    if (background.empty())
        return fgMask;

    cv::Mat grayFrame;
    cv::Mat grayBg;

    if (frame.channels() == 3)
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    else
        grayFrame = frame.clone();

    if (background.channels() == 3)
        cv::cvtColor(background, grayBg, cv::COLOR_BGR2GRAY);
    else
        grayBg = background.clone();

    cv::Mat frameF, bgF;
    grayFrame.convertTo(frameF, CV_32F);
    grayBg.convertTo(bgF, CV_32F);

    cv::Mat ratio = frameF / (bgF + 1.0);

    cv::Mat diff;
    cv::absdiff(frameF, bgF, diff);

    cv::Mat darker;
    cv::compare(frameF, bgF, darker, cv::CMP_LT);

    cv::Mat ratioHigh, ratioLow;
    cv::compare(ratio, 0.55, ratioHigh, cv::CMP_GT);
    cv::compare(ratio, 0.95, ratioLow, cv::CMP_LT);

    cv::Mat diffHigh, diffLow;
    cv::compare(diff, 5.0, diffHigh, cv::CMP_GT);
    cv::compare(diff, 80.0, diffLow, cv::CMP_LT);

    cv::Mat shadowMask;
    cv::bitwise_and(fgMask, darker, shadowMask);
    cv::bitwise_and(shadowMask, ratioHigh, shadowMask);
    cv::bitwise_and(shadowMask, ratioLow, shadowMask);
    cv::bitwise_and(shadowMask, diffHigh, shadowMask);
    cv::bitwise_and(shadowMask, diffLow, shadowMask);

    fgMask.setTo(0, shadowMask);

    return fgMask;
}
//---------------------------------------------------------------------------------------

    // creo filtro MOG2: separa il foreground da background. Ritorno il foreground come Mat

 std::vector<cv::Mat> Segmentation::filterMOG2(const std::vector<cv::Mat>& frames)
{
    std::vector<cv::Mat> foreground;

    std::ofstream logFile("motion_mask_stats.txt", std::ios::app);

    cv::Ptr<cv::BackgroundSubtractorMOG2> mog2 =
        cv::createBackgroundSubtractorMOG2(
            500,
            16.0,
            true
        );

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

        cv::Mat rawMask;
        mog2->apply(frame, rawMask);

        cv::Mat mask;

        // Tengo solo foreground vero, escludendo shadow = 127
        cv::inRange(rawMask, 254, 255, mask);

        int nonZero = cv::countNonZero(mask);

        double ratio =
            static_cast<double>(nonZero) /
            static_cast<double>(mask.total());

        logFile << "[MOG2] Frame: " << i
                << " | nonZero: " << nonZero
                << " | ratio: " << ratio
                << std::endl;

        bool lightChange = isLightChangeMask(mask, 0.35);

        if (lightChange)
        {
            logFile << "[MOG2] Frame: " << i
                    << " | LIGHT CHANGE DETECTED"
                    << std::endl;

            if (hasLastGoodMask)
            {
                foreground.push_back(lastGoodMask.clone());
            }
            else
            {
                foreground.push_back(cv::Mat::zeros(mask.size(), CV_8UC1));
            }

            continue;
        }

        if (nonZero > 0)
        {
            lastGoodMask = mask.clone();
            hasLastGoodMask = true;
        }

        foreground.push_back(mask);
    }

    logFile << "----------------------------------------" << std::endl;
    logFile.close();

    return foreground;
}

//---------------------------------------------------------------------------------------

//TROVA LA BBOX DA MASCHERA FINALE

std::vector<cv::Rect> Segmentation:: findBBox(
    const std::vector<cv::Mat>& masks,
    int minContourArea = 20)
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

        cv::threshold(mask, binary, 127, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(
            binary,
            contours,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE
        );

        if (contours.empty())
        {
            bboxes.push_back(cv::Rect());
            continue;
        }

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

// FUNZIONE PRINCIPALE DA GUARDARE !!
// funzione public che estrae le BBox finali 

void Segmentation::extractBBox( SequenceData& sequence) {

    std::vector<cv::Mat> filteredImgs; //vettore con immagini che vengo modificate e filtrate

// applico mog2 per ricavare foreground
    
    filteredImgs = (filterMOG2(sequence.frames)) ; 
    

//applico morphology
// NB: HO NOTATO CHE CON KERNEL (3,3) PRENDE ANCHE HANDCLAPPING, CON KERNEL (5,5) NON CI RIESCE
// IN QUESTO MODO NON TROVA USA FALLBACK, PER0' E' UN PO' PIU' IMPRECISO CON I CAMBI DI ILLUMINAZIONE
// NB: CAMBIO LA THRESHOLD PER EFFETTI SULL'ILLUMINAZIONE

    filteredImgs = (filterMorphology(filteredImgs));

// Salvo la motion mask (seeds) in sequence.motionMasks (utile per feature extraction)

    sequence.motionMasks = filteredImgs;

//  Decido se la sequenza è da MediaPipe

    bool useMediaPipeForSequence =
        shouldUseMediaPipeForSequence(filteredImgs, 0.80);

// Se NON è una sequenza da MediaPipe,
// allora riempio temporalmente i seed vuoti

    if (!useMediaPipeForSequence)
    {
        filteredImgs = fillEmptySeedsTemporal(filteredImgs);
    }       


// Cerco maschera fianle: applico GrabCut, se maschera ancora nera uso GrabcutMediaPipe

    std::vector<cv::Mat> finalImgsMask; 

    finalImgsMask = finalMask(
        sequence.frames,
        filteredImgs,
        sequence.name,
        useMediaPipeForSequence
        );

// Salvo nella struct sequences le final mask

    sequence.finalMasks = finalImgsMask;


//mostra se funziona maschera

    showMasks(finalImgsMask);  

// ritorna vettore di Rect

    std::vector<cv::Rect> bboxes = findBBox(finalImgsMask);

// Salvo nella struct sequences le BBoxes

    sequence.BBoxes = bboxes;

// SMOOTH BBOX
// lo smoothing non risponde benissimo quando i soggetti corrono

//    bboxes = smoothBBoxes(bboxes, 0.50, 3);

// mostra bbox

    showBBoxes(sequence.frames, bboxes);

}
//---------------------------------------------------------------------------------------