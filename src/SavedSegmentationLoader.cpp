/*
Matteo Gidoni
*/

#include "SavedSegmentationLoader.hpp"
#include "segmentation/BoundingBox.hpp"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <algorithm>

namespace fs = std::filesystem;

static bool loadMasksFromFolder(
    const fs::path& folder,
    std::vector<cv::Mat>& masks)
{
    masks.clear();

    if (!fs::exists(folder) || !fs::is_directory(folder))
    {
        std::cerr << "[SavedSegmentationLoader] Folder not found: "
                  << folder << std::endl;

        return false;
    }

    std::vector<cv::String> files;

    cv::glob((folder / "*.png").string(), files, false);

    std::vector<cv::String> jpgFiles;
    cv::glob((folder / "*.jpg").string(), jpgFiles, false);

    files.insert(files.end(), jpgFiles.begin(), jpgFiles.end());

    std::sort(files.begin(), files.end());

    for (const auto& filename : files)
    {
        cv::Mat mask =
            cv::imread(filename, cv::IMREAD_GRAYSCALE);

        if (!mask.empty())
        {
            masks.push_back(mask);
        }
    }

    // std::cout << "[SavedSegmentationLoader] Loaded "
    //           << masks.size()
    //           << " masks from: "
    //           << folder
    //           << std::endl;

    return !masks.empty();
}

//-------------------------------------------------------------------------------------------

bool loadSavedSegmentation(
    SequenceData& sequence,
    const std::string& resultsRoot)
{
    fs::path sequenceFolder = fs::path(resultsRoot) / sequence.name;

    fs::path finalMaskFolder = sequenceFolder / "masks";

    fs::path motionMaskFolder = sequenceFolder / "motion_masks";

    // std::cout
    //     << "[SavedSegmentationLoader] Looking for final masks in: "
    //     << finalMaskFolder
    //     << std::endl;

    bool finalMasksLoaded =
        loadMasksFromFolder(
            finalMaskFolder,
            sequence.finalMasks
        );

    // std::cout
    //     << "[SavedSegmentationLoader] Looking for motion masks in: "
    //     << motionMaskFolder
    //     << std::endl;

    bool motionMasksLoaded =
        loadMasksFromFolder(
            motionMaskFolder,
            sequence.motionMasks
        );

    if (!finalMasksLoaded)
    {
        std::cerr
            << "[SavedSegmentationLoader] Cannot load final masks for: "
            << sequence.name
            << std::endl;

        return false;
    }

    if (!motionMasksLoaded)
    {
        std::cerr
            << "[SavedSegmentationLoader] Cannot load motion masks for: "
            << sequence.name
            << std::endl;

        return false;
    }

    sequence.BBoxes =
        findBBox(sequence.finalMasks);

    if (sequence.finalMasks.size() !=
        sequence.motionMasks.size())
    {
        std::cerr
            << "[SavedSegmentationLoader] SIZE MISMATCH: finalMasks="
            << sequence.finalMasks.size()
            << " motionMasks="
            << sequence.motionMasks.size()
            << std::endl;

        return false;
    }

    if (sequence.BBoxes.size() !=
        sequence.finalMasks.size())
    {
        std::cerr
            << "[SavedSegmentationLoader] SIZE MISMATCH: BBoxes="
            << sequence.BBoxes.size()
            << " finalMasks="
            << sequence.finalMasks.size()
            << std::endl;

        return false;
    }

    return true;
}