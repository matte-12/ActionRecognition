#include "segmentation/ShadowRemoval.hpp"


// DARK MASK PER ELIMINARE LE OMBRE

// cv::Mat createDarkMask(const cv::Mat& frame) {
//     cv::Mat gray;

//     if (frame.channels() == 3)
//         cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
//     else
//         gray = frame.clone();

//     cv::Mat darkMask;

//     // Valore da tarare: 70/80/90 a seconda del dataset
//     cv::threshold(gray, darkMask, 80, 255, cv::THRESH_BINARY_INV);

//     return darkMask;
// }

