#include "segmentation/MediaPipeFallback.hpp"
#include <fstream>
#include <cstdlib>
#include <cmath>
//--------------------------------------------------------------------------------------

// FUNZIONE PER STABILIRE SE DOVREI USARE MEDIAPIPE O NO, IN BASE ALLA % DI SEEDS BUONI per cartella di farames

bool shouldUseMediaPipeForSequence(
    const std::vector<cv::Mat>& seeds,
    double zeroRatioThreshold )
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
        static_cast<double>(zeroCount) / static_cast<double>(seeds.size());

    return zeroRatio >= zeroRatioThreshold;
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