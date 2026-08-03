#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "Types.hpp"

class Segmentation {
public:
    void extractBBox( SequenceData& sequence);

// private: 
//     std::vector<cv::Mat> filterMOG2(const std::vector<cv::Mat>& frames); // filtra con Mog2
//     std::vector<cv::Mat> filterMorphology(const std::vector<cv::Mat>& frames);// filtra con morfologia
//     std::vector<cv::Mat> finalMask(
//         const std::vector<cv::Mat>& frames, 
//         const std::vector<cv::Mat>& seeds,
//         const std::string& sequenceName,
//         const bool useMediaPipeForSequence
//     );
//     std::vector<cv::Rect> findBBox(
//     const std::vector<cv::Mat>& masks,
//     int minContourArea); 

};
