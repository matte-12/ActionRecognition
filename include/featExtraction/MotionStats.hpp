#pragma once

#include "Types.hpp"

struct MotionStats
{
    double motionEnergy = 0.0;

    double upper = 0.0;
    double middle = 0.0;
    double lower = 0.0;

    double left = 0.0;
    double right = 0.0;

    double centerRatio = 0.0;
};

MotionStats computeBodyMotion(
    const std::vector<cv::Mat>& motionMasks,
    const std::vector<cv::Rect>& boxes);