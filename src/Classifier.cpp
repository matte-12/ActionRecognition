#include "Classifier.hpp"

#include <cmath>
#include <iostream>

// ------------------------------------------------------------
// Label mapping
// 1= Boxing 2= handclapping 3= handwaving 4= jogging 5= running 6= walking

// ------------------------------------------------------------


static bool isLocomotion(const SequenceFeatures& features)
{
    return features.motionEnergySeed > 0.09;
}


static int classifyLocomotion(const SequenceFeatures& features)
{
    // displacement basso -> generalmente walking
    if (features.displacementNorm <= 1.11)
    {
        // eccezione: molto movimento nella zona centrale
        // tende più verso jogging
        if (features.middleMotion > 0.28)
        {
            return 4; // jogging
        }

        return 6; // walking
    }

    // displacement alto:
    // velocity distingue abbastanza bene jogging/running
    if (features.velocityNorm <= 0.06)
    {
        return 4; // jogging
    }

    return 5; // running
}


static int classifyStaticAction(const SequenceFeatures& features)
{
    /*
        Prima separazione:
        boxing tende ad avere aspectRatioVariation più bassa.
    */
    if (features.aspectRatioVariation <= 0.10)
    {
        /*
            Se la variazione geometrica è piccola ma anche
            upperMotion è praticamente nullo, tende a essere
            handclapping.
        */
        if (features.upperMotion <= 0.01)
        {
            return 2; // handclapping
        }

        return 1; // boxing
    }

    /*
        Aspect ratio più variabile:
        guardiamo dove si concentra il movimento.

        Handwaving tende ad avere movimento più laterale,
        quindi centerMotionRatio più basso.
    */
    if (features.centerMotionRatio <= 0.17)
    {
        return 3; // handwaving
    }

    return 2; // handclapping
}


int Classifier::predict(const SequenceFeatures& features)
{
    int predictedLabel;

    if (isLocomotion(features))
    {
        predictedLabel =
            classifyLocomotion(features);
    }
    else
    {
        predictedLabel =
            classifyStaticAction(features);
    }


    return predictedLabel;
}