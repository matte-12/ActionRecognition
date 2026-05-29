#include "Metrics.hpp"

double Metrics::computeIoU(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect intersection = a & b;
    double interArea = intersection.area();
    double unionArea = a.area() + b.area() - interArea;

    if (unionArea <= 0.0) {
        return 0.0;
    }

    return interArea / unionArea;
}
