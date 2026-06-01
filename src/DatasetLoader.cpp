#include "DatasetLoader.hpp"


// funzione privata che divide il rootpath data/sequences nelle sottocartelle

std::vector<std::string> DatasetLoader:: getSubfolderPath(const std::string& rootPath){

    std::vector<std::string> folders;
    namespace fs = std::filesystem;

    for (const auto& entry : fs::directory_iterator(rootPath)) {
        if (entry.is_directory()) {
            folders.push_back(entry.path().string());
        }
    }

    return folders;
}


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
//
//mancano gtBox20 e label
//
    sequenza.push_back(seq);
}


    return sequenza;
}
