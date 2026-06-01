#include <opencv2/opencv.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

namespace fs = std::filesystem;

std::vector<std::string> getImageFiles(const std::string& sequencePath) {
    std::vector<std::string> imageFiles;

    for (const auto& entry : fs::directory_iterator(sequencePath)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
                imageFiles.push_back(entry.path().string());
            }
        }
    }

    std::sort(imageFiles.begin(), imageFiles.end());
    return imageFiles;
}

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

int inferClassIdFromName(const std::string& sequenceName) {
    std::string name = toLower(sequenceName);

    std::cout << "DEBUG sequence name: " << name << std::endl;

    if (name.find("walking") != std::string::npos) return 1;
    if (name.find("jogging") != std::string::npos) return 2;
    if (name.find("running") != std::string::npos) return 3;
    if (name.find("boxing") != std::string::npos) return 4;

    if (name.find("handwaving") != std::string::npos ||
        name.find("hand_waving") != std::string::npos ||
        name.find("hand waving") != std::string::npos) return 5;

    if (name.find("handclapping") != std::string::npos ||
        name.find("hand_clapping") != std::string::npos ||
        name.find("hand clapping") != std::string::npos) return 6;

    return -1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./annotate_gt <dataset_path>" << std::endl;
        return 1;
    }

    std::string datasetPath = argv[1];

    for (const auto& sequenceEntry : fs::directory_iterator(datasetPath)) {
        if (!sequenceEntry.is_directory())
            continue;

        std::string sequencePath = sequenceEntry.path().string();
        std::string sequenceName = sequenceEntry.path().filename().string();

        int classId = inferClassIdFromName(sequenceName);

        if (classId < 0) {
            std::cout << "Could not infer class id for sequence: "
                      << sequenceName << std::endl;

            std::cout << "Insert class id manually:" << std::endl;
            std::cout << "1=walking, 2=jogging, 3=running, 4=boxing, 5=handwaving, 6=handclapping" << std::endl;
            std::cin >> classId;
        }

        std::vector<std::string> imageFiles = getImageFiles(sequencePath);

        if (imageFiles.size() < 20) {
            std::cout << "Skipping " << sequenceName
                      << " (not enough frames)" << std::endl;
            continue;
        }

        std::string frame20Path = imageFiles[19];

        cv::Mat image = cv::imread(frame20Path);

        if (image.empty()) {
            std::cout << "Could not load " << frame20Path << std::endl;
            continue;
        }

        std::cout << "\nSequence: " << sequenceName << std::endl;
        std::cout << "Detected class id: " << classId << std::endl;
        std::cout << "Frame: " << frame20Path << std::endl;
        std::cout << "Draw bounding box around the full person and press ENTER." << std::endl;

        cv::Rect roi = cv::selectROI(sequenceName, image);

        if (roi.width <= 0 || roi.height <= 0) {
            std::cout << "Skipped." << std::endl;
            cv::destroyAllWindows();
            continue;
        }

        int xCenter = roi.x + roi.width / 2;
        int yCenter = roi.y + roi.height / 2;

        std::ofstream gtFile(sequencePath + "/gt.txt");

        gtFile << classId << " "
               << xCenter << " "
               << yCenter << " "
               << roi.width << " "
               << roi.height;

        gtFile.close();
std::cout << "Class ID before saving: " << classId << std::endl;
        std::cout << "Saved gt.txt: "
                  << classId << " "
                  << xCenter << " "
                  << yCenter << " "
                  << roi.width << " "
                  << roi.height << std::endl;

        cv::destroyAllWindows();
    }

    std::cout << "\nAll annotations completed." << std::endl;

    return 0;
}
