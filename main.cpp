#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include "DatasetLoader.hpp"
#include "segmentation/Segmentation.hpp"
#include "Visualizer.hpp"
#include "Classifier.hpp"
#include "FeatureExtractor.hpp"
#include "Metrics.hpp"

//----------------------------------------------------------------------------------------
//FUNZIONI DEL MAIN
//-------------------------------------------------------------------------------------------



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
//-------------------------------------------------------------------------------------

// da fare...


//-------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------
// passo da DatasetLoader per creare vettore di SequenceData
// cioè prendo i frame da data/sequences o local_sequences (cartella locale completa)
// ottengo elementi del tipo struct SequenceData
//-------------------------------------------------------------------------------------

    std::string sequencesPath = argv[1];
    //std::string sequencesPath = "local_sequences/";

    DatasetLoader loader;
    std::vector<SequenceData> sequenze = loader.loadDataset(sequencesPath);
        
//-------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------
// carico tutti i SequenceData.frames, uno alla volta, in segmentation. 
//
// NB: SALVO I BBOXES OTTENUTI IN seq.BBoxes. Le mask finali in seq.finalMask.
//  le motion masks in seq.motionMasks (potrebbe essere utile per feature extraction)!!! 
//
// mostra le mask e bbox in tempo reale
//-------------------------------------------------------------------------------------
    
    Segmentation segment;

    //seq ausiliario per usare Segmentation su tutti i SequenceData sequenze
    for( auto& seq : sequenze){
        segment.extractBBox(seq);
    }


//-------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------
//Visualizer va messo per ultimo. Servono le predictedLabel estratte con FeatureExtraction
// e classificate con Classifier!!

//NB int predictedLabel: 1= Walking 2= jogging ... (da decidere)
        // case 1: return "walking";
        // case 2: return "jogging";
        // case 3: return "running";
        // case 4: return "boxing";
        // case 5: return "handwaving";
        // case 6: return "handclapping";
        // default: return "unknown";

// Salvo frame unito  BBox unito predictedLabel nella cartella results/ 

// DA MODIFICARE PREDICTEDLABEL !!!
//-------------------------------------------------------------------------------------

    int predictedLabel = 1; // messo di prova

    std::string resultsPath = argv[2];
    //std::string resultsPath = "results/";

    Visualizer visualizer;

    for (const auto& seq : sequenze){

        visualizer.saveSequenceOutput(
            seq,
            predictedLabel,
            resultsPath
        );

    }
    

//---------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------

    return 0;
}
