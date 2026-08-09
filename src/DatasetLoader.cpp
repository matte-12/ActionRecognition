/*
Matteo Gidoni
*/

#include "DatasetLoader.hpp"
#include <fstream>

namespace fs = std::filesystem;

//---------------------------------------------------------------------------------------------

// IMPOSTA LA SesquenceData sequence .trueLabel, .gtBBox20, .hasGT.
// ricava questi valori dalla groundtruth.txt

bool DatasetLoader::getGroundTruth( SequenceData& sequence ) {

    auto datasetPath = sequence.name;
    std::string gtPath = datasetPath + "/labels/ground_truth.txt";

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
    double xCenter;
    double yCenter;
    double width;
    double height;

    file >> classId >> xCenter >> yCenter >> width >> height;

    if (file.fail()) {
        std::cerr << "Invalid gt format in: " << gtPath << std::endl;
        std::cerr << "Expected: <class_id> <x_center> <y_center> <width> <height>" << std::endl;
        sequence.hasGT = false;
        return false;
    }

    int x = static_cast<int>(std::round(xCenter - width / 2.0));
    int y = static_cast<int>(std::round(yCenter - height / 2.0));
    int w = static_cast<int>(std::round(width));
    int h = static_cast<int>(std::round(height));

    sequence.trueLabel = classId;
    sequence.gtBBox20 = cv::Rect(x, y, w, h);
    sequence.hasGT= true;

    return true;
}


//---------------------------------------------------------------------------------------------

// restituisce il nome del path fino a prima di data e labels

std::vector<std::string> DatasetLoader::getSubfolderPath(const std::string& rootPath)
{
    std::vector<std::string> sequenceFolders;

    for (const auto& entry : fs::recursive_directory_iterator(rootPath))
    {
        if (!entry.is_directory())
            continue;

        fs::path currentPath = entry.path();

        fs::path dataPath = currentPath / "data";
        fs::path labelsPath = currentPath / "labels";

        if (fs::exists(dataPath) && fs::is_directory(dataPath) &&
            fs::exists(labelsPath) && fs::is_directory(labelsPath))
        {
            sequenceFolders.push_back(currentPath.string());
        }
    }

    return sequenceFolders;
}

//---------------------------------------------------------------------------------------------

// RICAVO I FRAME DALLA CARTELLA

std::vector<cv::Mat> DatasetLoader :: getFrames(const std::string& datasetPath)
{
    // entro nella cartella data
    auto data = datasetPath + "/data";

    std::vector<cv::String> filenames;
    std::vector<cv::Mat> images;

    // Prende tutti i file jpg 
    cv::glob(data + "/*.jpg", filenames, false);
    
    // mette i png:
    std::vector<cv::String> pngFiles;
    cv::glob(data + "/*.png", pngFiles, false);

    // unisce tutto
    filenames.insert(filenames.end(), pngFiles.begin(), pngFiles.end());
    std::sort(filenames.begin(), filenames.end());

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

    // creo un vettore contenente le sottocartelle di local_sequences/
    auto subfolderPath =  getSubfolderPath(rootPath); 

    //imposto gli elementi della stucture: .name, .frames, .gtBBox20
    for (const auto& path : subfolderPath)
{

    SequenceData seq;

    seq.name = path;
    seq.frames = getFrames(path);
    

// funzione che imposta .trueLabel, .gtBBox20, .hasGT. Sempre che sia presente la gt.txt

    bool flag = DatasetLoader:: getGroundTruth(seq);
    if(!flag){
        std::cout<< "subfolder"<< seq.name << "non ha caricato correttamente la groundTruth" << std::endl;
    }

    sequenza.push_back(seq);

}


    return sequenza;
}
