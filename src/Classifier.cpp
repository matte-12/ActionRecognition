#include "Classifier.hpp"

#include <cmath>
#include <iostream>

// ------------------------------------------------------------
// Label mapping
// 1= Boxing 2= handclapping 3= handwaving 4= jogging 5= running 6= walking

// ------------------------------------------------------------


static bool isLocomotion(const SequenceFeatures& features)
{
    return std::abs(features.totalDisplacementX) > 40.0 ||
           features.totalDisplacement > 45.0;
}

static int classifyLocomotion(const SequenceFeatures& features)
{
    if (features.meanVelocity < 2.0)
        return 1; // walking

    if (features.meanVelocity < 5.0)
        return 2; // jogging

    return 3; // running
}


// ------------------------------------------------------------

static int classifyStaticAction(const SequenceFeatures& features)
{
    bool upperDominant =
        features.upperMotion > features.middleMotion &&
        features.upperMotion > features.lowerMotion;

    bool middleDominant =
        features.middleMotion >= features.upperMotion &&
        features.middleMotion >= features.lowerMotion;

    bool highAreaVariation =
        features.meanArea > 0.0 &&
        features.areaVariation > features.meanArea * 0.25;

    bool highAspectRatio =
        features.meanAspectRatio > 0.75;

    if (upperDominant && highAreaVariation)
        return 5; // handwaving

    if (upperDominant && !highAreaVariation)
        return 6; // handclapping

    if (middleDominant || highAspectRatio)
        return 4; // boxing

    return 4; // fallback
}

// ------------------------------------------------------------


int Classifier::predict(const SequenceFeatures& features)
{
    if (isLocomotion(features))
        return classifyLocomotion(features);

    return classifyStaticAction(features);
}