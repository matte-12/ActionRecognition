#pragma once

#include "Types.hpp"

struct OpticalFlow
{
    double meanFlowMagnitude = 0.0;
    double maxFlowMagnitude = 0.0;

    double upperHorizontalFlow = 0.0;
    double upperVerticalFlow = 0.0;

    double flowConvergence = 0.0;

    double lowerFlowMagnitude = 0.0;

    double leftX = 0.0;
    double rightX = 0.0;

    double leftAbsX = 0.0;
    double rightAbsX = 0.0;

    double horizontalSymmetry = 0.0;

    double strideFrequency = 0.0;
};

OpticalFlow computeOpticalFlow(const SequenceData& sequence);