#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>

#include "DatasetLoader.hpp"
#include "segmentation/Segmentation.hpp"
#include "Visualizer.hpp"
#include "Classifier.hpp"
#include "FeatureExtractor.hpp"
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
//controlli sull'argv -------------------------------------------------------------------

// da fare...


//-------------------------------------------------------------------------------------

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
//
// NB: SALVO I BBOXES OTTENUTI IN seq.BBoxes. Le mask finali in seq.finalMask.
//  le motion masks in seq.motionMasks (potrebbe essere utile per feature extraction)!!! 
//
// mostra le mask e bbox in tempo reale
//-------------------------------------------------------------------------------------
 

//-------------------------------------------------------------------------------------
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
    std::vector<int> predictedLabels;

    Metrics metrics;

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

        int predictedLabel = classifier.predict(features);
        predictedLabels.push_back(predictedLabel);

        featureExt.logFeatures(features, "features_log.csv");

        metrics.addResult(seq, predictedLabel);

        validSequences.push_back(&seq);
    }
        

//-------------------------------------------------------------------------------------
//NB int predictedLabel: 1= Boxing 2= handclapping 3= handwaving 4= jogging 5= running 6= walking

// Salvo frame unito  BBox unito predictedLabel nella cartella results/ 
//-------------------------------------------------------------------------------------
    if(!haveSegments){
        Visualizer visualizer;

        for (size_t i = 0; i < validSequences.size(); i++)
        {
            visualizer.saveSequenceOutput(
                *validSequences[i],
                predictedLabels[i],
                resultsPath
            );
        }
    }
//-------------------------------------------------------------------------------------
    // Metrics

    metrics.printSummary();
    metrics.saveResultsCSV(resultsPath + "/classification_results.csv");
    metrics.saveSummaryCSV(resultsPath + "/metrics_summary.txt");

//-------------------------------------------------------------------------------------

    return 0;
}
