#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "Types.hpp"
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace fs = std::filesystem;

class Visualizer {
public:
    void saveSequenceOutput(
        const SequenceData& sequenza,
        int predictedLabel,
        const std::string& outputFolder
    );

private: 
    std::string conversione (int predictedLabel); // converte da int di predLabel a stringa ("walking", ...)
    cv::Mat unione(cv::Mat& frame, std::string& label, cv::Rect& bbox); // unisce frame,bbox, label
    void salvataggio (cv:: Mat& img, fs::path& sequenceOutputPath, int counter);
    void saveMasks(
    const std::string& sequenceName,
    const std::vector<cv::Mat>& sequence,
    const std::string& outputFolder,
    const std::string lastFolderName
    );
};
