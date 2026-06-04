#include "DatasetLoader.hpp"
#include <fstream>

namespace fs = std::filesystem;

//---------------------------------------------------------------------------------------------

// IMPOSTA LA SesquenceData sequence .trueLabel, .gtBBox20, .hasGT.
// ricava questi valori dalla groundtruth.txt

bool DatasetLoader::getGroundTruth(const std::string& datasetPath, SequenceData& sequence) {
    std::string gtPath = datasetPath + "/gt.txt";

    if (!fs::exists(gtPath)) {
        sequence.hasGT= false;
        return false;
    }

    std::ifstream file(gtPath);

    if (!file.is_open()) {
        std::cerr << "Could not open gt file: " << gtPath << std::endl;
        sequence.hasGT = false;
        return false;
    }

    int classId;
    int xCenter;
    int yCenter;
    int width;
    int height;

    file >> classId >> xCenter >> yCenter >> width >> height;

    if (file.fail()) {
        std::cerr << "Invalid gt format in: " << gtPath << std::endl;
        std::cerr << "Expected: <class_id> <x_center> <y_center> <width> <height>" << std::endl;
        sequence.hasGT = false;
        return false;
    }

    int x = xCenter - width / 2;
    int y = yCenter - height / 2;

    sequence.trueLabel = classId;
    sequence.gtBBox20 = cv::Rect(x, y, width, height);
    sequence.hasGT= true;

    return true;
}


//---------------------------------------------------------------------------------------------

// funzione privata che divide il rootpath data/sequences nelle sottocartelle

std::vector<std::string> DatasetLoader:: getSubfolderPath(const std::string& rootPath){

    std::vector<std::string> folders;

    for (const auto& entry : fs::directory_iterator(rootPath)) {
        if (entry.is_directory()) {
            folders.push_back(entry.path().string());
        }
    }

    return folders;
}

//---------------------------------------------------------------------------------------------

// RICAVO I FRAME DALLA CARTELLA

std::vector<cv::Mat> DatasetLoader :: getFrames(const std::string& datasetPath)
{
    std::vector<cv::String> filenames;
    std::vector<cv::Mat> images;

    // Prende tutti i file jpg 
    cv::glob(datasetPath + "/*.jpg", filenames, false);
    
    // mette i png:
    std::vector<cv::String> pngFiles;
    cv::glob(datasetPath + "/*.png", pngFiles, false);

    // unisce tutto
    filenames.insert(filenames.end(), pngFiles.begin(), pngFiles.end());

    // Carica le immagini + le converto in grigi
    for (const auto& file : filenames)
    {
        cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);
        if (!img.empty())
        {
            images.push_back(img);
        }
    }

    return images;
}

//---------------------------------------------------------------------------------------------

// METODO PUBLIC PRINCIPALE

// devo restituire un vettore di SequenceData (structure definita in Types)

std::vector<SequenceData> DatasetLoader::loadDataset(const std::string& rootPath) {
    std::vector<SequenceData> sequenza;
    auto subfolderPath =  getSubfolderPath(rootPath); // creo un vettore contenente le sottocartelle di data/sequences

    //imposto gli elementi della stucture: .name, .frames, .gtBBox20
    for (const auto& path : subfolderPath)
{

    SequenceData seq;

    seq.name = path;
    seq.frames = getFrames(path);
    sequenza.push_back(seq);

// funzione che imposta .trueLabel, .gtBBox20, .hasGT. Sempre che sia presente la gt.txt

    bool flag = DatasetLoader:: getGroundTruth(seq.name, seq);
    if(!flag){
        std::cout<< "subfolder"<< seq.name << "non ha caricato correttamente la groundTruth" << std::endl;
    }

}


    return sequenza;
}
