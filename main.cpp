#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

#include "DatasetLoader.hpp"
#include "segmentation/Segmentation.hpp"
#include "Visualizer.hpp"
#include "Classifier.hpp"
#include "featExtraction/FeatureExtractor.hpp"
#include "Metrics.hpp"
#include "SavedSegmentationLoader.hpp"


//----------------------------------------------------------------------------------------
//FUNZIONI DEL MAIN
//-------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------

int main(int argc, char** argv) {


    std::cout << "Action Recognition: " << std::endl;

    if (argc >= 3) {
        std::cout << "Dataset path: " << argv[1] << std::endl;
        std::cout << "Results path: " << argv[2] << std::endl;
    } else {
        std::cout << "Usage: ./ActionRecognition <dataset_path> <results_path>" << std::endl;
    }


//-------------------------------------------------------------------------------------
// passo da DatasetLoader per creare vettore di SequenceData
// cioè prendo i frame da data/sequences o local_sequences (cartella locale completa)
// ottengo elementi del tipo struct SequenceData
//-------------------------------------------------------------------------------------

    std::string sequencesPath = argv[1];
    //std::string sequencesPath = "local_sequences/";

    std::string resultsPath = argv[2];
    //std::string resultsPath = "results/";

    DatasetLoader loader;
    std::vector<SequenceData> sequenze = loader.loadDataset(sequencesPath);

//-------------------------------------------------------------------------------------
// carico tutti i SequenceData.frames, uno alla volta, in segmentation. 
// Segmentation / reload --> Feature extraction
// Feature Extractor + non ricalcola le maschere con flag + log file

    std::ofstream clearFeatureLog("features_log.csv", std::ios::trunc);
    clearFeatureLog.close();

    std::ofstream clearMotionLog("motion_mask_stats.txt", std::ios::trunc);
    clearMotionLog.close();
   

    Segmentation segment;

    // trick per decidere se ricalcolare le mask o meno
    bool haveSegments = false;

    FeatureExtractor featureExt;
    std::vector<SequenceFeatures> allFeatures;
    std::vector<SequenceData*> validSequences;

    Classifier classifier;

    for (auto& seq : sequenze)
    {
        if (!haveSegments)
        {
            segment.extractBBox(seq);
        }
        else
        {
            bool loaded = loadSavedSegmentation(seq, resultsPath);

            if (!loaded)
            {
                std::cerr << "Could not load saved segmentation for: "
                        << seq.name << std::endl;
                continue;
            }
        }

        SequenceFeatures features = featureExt.extract(seq);

        allFeatures.push_back(features);

        featureExt.logFeatures(features, "features_log.csv");

        validSequences.push_back(&seq);
    }



//-------------------------------------------------------------------------------------
//
// LEAVE-ONE-SUBJECT-OUT
//
// Fold 1: train person02 + person03 -> test person01
// Fold 2: train person01 + person03 -> test person02
// Fold 3: train person01 + person02 -> test person03
//

std::vector<std::string> testPersons = {
    "person01",
    "person02",
    "person03"
};

Metrics metrics;

std::vector<int> allPredictedLabels(
    allFeatures.size(),
    -1
);


// ---------------------
// Esegui i 3 fold
// ---------------------


for (const std::string& testPerson : testPersons)
{
    std::cout
        << "\n===================================="
        << std::endl;

    std::cout
        << "[Main] Test subject: "
        << testPerson
        << std::endl;

    std::cout
        << "===================================="
        << std::endl;


    std::vector<SequenceFeatures> trainFeatures;
    std::vector<size_t> testIndices;


    // ---------------------
    // Split train / test
    // ---------------------

    for (size_t i = 0; i < allFeatures.size(); i++)
    {
        const SequenceFeatures& features = allFeatures[i];

        if (features.sequenceName.find(testPerson)!= std::string::npos)
        {
            testIndices.push_back(i);
        }
        else
        {
            trainFeatures.push_back(features);
        }
    }


    std::cout
        << "[Main] Training samples: "
        << trainFeatures.size()
        << std::endl;

    std::cout
        << "[Main] Test samples: "
        << testIndices.size()
        << std::endl;


    if (trainFeatures.empty() ||testIndices.empty())
    {
        std::cerr
            << "[Main] Invalid split for "
            << testPerson
            << std::endl;

        continue;
    }


    // -------------------------------
    // Nuovo classifier per ogni fold
    // -------------------------------

    Classifier classifier;

    classifier.train(trainFeatures);


    // ----------------------------
    // Test
    // ----------------------------

    int foldCorrect = 0;


    for (size_t index : testIndices)
    {
        const SequenceFeatures& features = allFeatures[index];

        SequenceData& sequence = *validSequences[index];


        int predictedLabel = classifier.predict(features);


        allPredictedLabels[index] = predictedLabel;


        if (predictedLabel == features.trueLabel)
        {
            foldCorrect++;
        }


        // IMPORTANTE:
        // accumuliamo tutte le sequenze di test
        // dentro lo stesso Metrics
        metrics.addResult(
            sequence,
            predictedLabel
        );
    }


    double foldAccuracy = static_cast<double>(foldCorrect) /static_cast<double>(testIndices.size());


    std::cout
        << "[Main] Accuracy "
        << testPerson
        << ": "
        << foldAccuracy
        << std::endl;
}


//-------------------------------------------------------------------------------------

// METRICHE AGGREGATE
//
// Ogni sequenza è stata usata come test una sola volta.


std::cout << "\n\n====================================" << std::endl;

std::cout << "LOSO METRICS" << std::endl;

std::cout << "====================================" << std::endl;


metrics.printSummary();

metrics.saveResultsCSV(resultsPath+ "/classification_results_loso.csv");

metrics.saveSummaryCSV(resultsPath  + "/metrics_summary_loso.csv");


//-------------------------------------------------------------------------------------
// Visualizzazione
//
// Ora ogni sequenza ha una predizione ottenuta
// quando quella persona era nel test set.

    if(!haveSegments){
        Visualizer visualizer;

        for (size_t i = 0; i < validSequences.size(); i++)
        {
            if (allPredictedLabels[i] == -1)
                continue;

            visualizer.saveSequenceOutput(
                *validSequences[i],
                allPredictedLabels[i],
                resultsPath
            );
        }
    }
//-------------------------------------------------------------------------------------

    return 0;
}
