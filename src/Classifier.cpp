#include "Classifier.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

#include <iostream>
#include <vector>


//------------------------------------------------------------
// ------------------------------------------------------------
// Label mapping
// 1= Boxing
// 2= handclapping 
// 3= handwaving 
// 4= jogging
// 5= running
// 6= walking
//------------------------------------------------------------
//------------------------------------------------------------


//------------------------------------------------------------
// trasformare vettori in matrici 1xN

static cv::Mat vectorToRow(
    const std::vector<float>& values)
{
    cv::Mat row(
        1,
        static_cast<int>(values.size()),
        CV_32F
    );

    for (size_t i = 0; i < values.size(); i++)
    {
        row.at<float>(0, static_cast<int>(i)) =
            values[i];
    }

    return row;
}
//------------------------------------------------------------

static std::vector<float> getLocomotionFeatureVector(
    const SequenceFeatures& f)
{
    return {
        static_cast<float>(f.displacementNorm),
        static_cast<float>(f.pathLengthNorm),

        static_cast<float>(f.velocityNorm),
        static_cast<float>(f.trajectoryEfficiency),

        static_cast<float>(f.motionEnergySeed),

        static_cast<float>(f.meanFlowMagnitude),
        static_cast<float>(f.maxFlowMagnitude),
        static_cast<float>(f.lowerFlowMagnitude),

        static_cast<float>(f.strideFrequency)
    };
}
//------------------------------------------------------------

static std::vector<float> getStaticFeatureVector(const SequenceFeatures& f)
{
    return {
        static_cast<float>(f.aspectRatioVariation),
        static_cast<float>(f.widthVariationNorm),

        static_cast<float>(f.upperMotion),
        static_cast<float>(f.middleMotion),
        static_cast<float>(f.centerMotionRatio),

        static_cast<float>(f.upperHorizontalFlow),
        static_cast<float>(f.upperVerticalFlow),

        static_cast<float>(f.flowConvergence),

        static_cast<float>(f.leftAbsX),
        static_cast<float>(f.rightAbsX),
        static_cast<float>(f.horizontalSymmetry)
    };
}

//------------------------------------------------------------
// separa locomotion / non-locomotion
static bool isLocomotion(const SequenceFeatures& features)
{
    return features.motionEnergySeed > 0.09;
}

//------------------------------------------------------------
// static int classifyLocomotion(const SequenceFeatures& features)
// {
//     // displacement basso -> generalmente walking
//     if (features.displacementNorm <= 1.11)
//     {
//         // eccezione: molto movimento nella zona centrale
//         // tende più verso jogging
//         if (features.middleMotion > 0.28)
//         {
//             return 4; // jogging
//         }

//         return 6; // walking
//     }

//     // displacement alto:
//     // velocity distingue abbastanza bene jogging/running
//     if (features.velocityNorm <= 0.06)
//     {
//         return 4; // jogging
//     }

//     return 5; // running
// }

// //------------------------------------------------------------

// static int classifyStaticAction(const SequenceFeatures& features)
// {
//     /*
//         Prima separazione:
//         boxing tende ad avere aspectRatioVariation più bassa.
//     */
//     if (features.aspectRatioVariation <= 0.10)
//     {
//         /*
//             Se la variazione geometrica è piccola ma anche
//             upperMotion è praticamente nullo, tende a essere
//             handclapping.
//         */
//         if (features.upperMotion <= 0.01)
//         {
//             return 2; // handclapping
//         }

//         return 1; // boxing
//     }

//     /*
//         Aspect ratio più variabile:
//         guardiamo dove si concentra il movimento.

//         Handwaving tende ad avere movimento più laterale,
//         quindi centerMotionRatio più basso.
//     */
//     if (features.centerMotionRatio <= 0.17)
//     {
//         return 3; // handwaving
//     }

//     return 2; // handclapping
// }

//------------------------------------------------------------
// creo matrice + labels

static void buildTrainingMatrices(
    const std::vector<std::vector<float>>& samples,
    const std::vector<int>& labels,
    cv::Mat& data,
    cv::Mat& responses)
{
    if (samples.empty())
        return;

    int rows = static_cast<int>(samples.size());

    int cols = static_cast<int>(samples[0].size());

    data = cv::Mat(
        rows,
        cols,
        CV_32F
    );

    responses = cv::Mat(
        rows,
        1,
        CV_32S
    );

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            data.at<float>(i, j) = samples[i][j];
        }

        responses.at<int>(i, 0) = labels[i];
    }
}

//------------------------------------------------------------
// configurazione random forest

static void configureForest(cv::Ptr<cv::ml::RTrees>& forest)
{
    forest->setMaxDepth(8);

    forest->setMinSampleCount(2);

    forest->setRegressionAccuracy(0.0f);

    forest->setUseSurrogates(false);

    forest->setMaxCategories(3);

    forest->setCalculateVarImportance(true);

    // 0 = OpenCV sceglie automaticamente
    // quante feature provare a ogni split
    forest->setActiveVarCount(0);

    forest->setTermCriteria(
        cv::TermCriteria(
            cv::TermCriteria::MAX_ITER,
            100,
            0
        )
    );
}

//------------------------------------------------------------
// training

void Classifier::train(const std::vector<SequenceFeatures>& trainingFeatures)
{
    trained = false;

    // Static actions

    std::vector<std::vector<float>> staticSamples;

    std::vector<int> staticLabels;


    // Locomotion

    std::vector<std::vector<float>> locomotionSamples;

    std::vector<int> locomotionLabels;


    // Dividi i campioni nei due gruppi

    for (const SequenceFeatures& f : trainingFeatures)
    {
        if (f.trueLabel < 1 || f.trueLabel > 6)
        {
            continue;
        }


        if (isLocomotion(f))
        {
            locomotionSamples.push_back(getLocomotionFeatureVector(f));

            locomotionLabels.push_back(f.trueLabel);
        }
        else
        {
            staticSamples.push_back(getStaticFeatureVector(f));

            staticLabels.push_back(f.trueLabel);
        }
    }


    // Controllo dati

    if (staticSamples.empty())
    {
        std::cerr
            << "[Classifier] No static training samples."
            << std::endl;

        return;
    }

    if (locomotionSamples.empty())
    {
        std::cerr
            << "[Classifier] No locomotion training samples."
            << std::endl;

        return;
    }


    // Crea matrici OpenCV

    cv::Mat staticData;
    cv::Mat staticResponses;

    buildTrainingMatrices(
        staticSamples,
        staticLabels,
        staticData,
        staticResponses
    );


    cv::Mat locomotionData;
    cv::Mat locomotionResponses;

    buildTrainingMatrices(
        locomotionSamples,
        locomotionLabels,
        locomotionData,
        locomotionResponses
    );


    // Training Random Forest statiche
    

    bool staticOk =
        staticForest->train(
            staticData,
            cv::ml::ROW_SAMPLE,
            staticResponses
        );


    // Training Random Forest locomotion

    bool locomotionOk =
        locomotionForest->train(
            locomotionData,
            cv::ml::ROW_SAMPLE,
            locomotionResponses
        );


    trained = staticOk && locomotionOk;


    // Debug

    std::cout
        << "[Classifier] Static samples: "
        << staticSamples.size()
        << std::endl;

    std::cout
        << "[Classifier] Locomotion samples: "
        << locomotionSamples.size()
        << std::endl;

    std::cout
        << "[Classifier] Training: "
        << (trained ? "OK" : "FAILED")
        << std::endl;
}

//------------------------------------------------------------
//costruttore

Classifier::Classifier()
{
    staticForest = cv::ml::RTrees::create();

    locomotionForest = cv::ml::RTrees::create();

    configureForest(staticForest);
    configureForest(locomotionForest);
}

//------------------------------------------------------------
// funzione main


int Classifier::predict(
    const SequenceFeatures& features) const
{
    if (!trained)
    {
        std::cerr
            << "[Classifier] Model not trained."
            << std::endl;

        return -1;
    }


    // Locomotion

    if (isLocomotion(features))
    {
        std::vector<float> values = getLocomotionFeatureVector(features);

        cv::Mat sample = vectorToRow(values);

        float prediction = locomotionForest->predict(sample);

        int label = static_cast<int> (std::round(prediction));

        // std::cout
        //     << "[Classifier] "
        //     << features.sequenceName
        //     << " | group=locomotion"
        //     << " | true=" << features.trueLabel
        //     << " | predicted=" << label
        //     << std::endl;

        return label;
    }


    // Static action

    std::vector<float> values = getStaticFeatureVector(features);

    cv::Mat sample = vectorToRow(values);

    float prediction = staticForest->predict(sample);

    int label = static_cast<int> (std::round(prediction));
    
    // std::cout
    //     << "[Classifier] "
    //     << features.sequenceName
    //     << " | group=static"
    //     << " | true=" << features.trueLabel
    //     << " | predicted=" << label
    //     << std::endl;

    return label;
}