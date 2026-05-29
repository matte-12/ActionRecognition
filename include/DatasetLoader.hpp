#pragma once

#include "Types.hpp"
#include <string>
#include <vector>

class DatasetLoader {
public:
    std::vector<SequenceData> loadDataset(const std::string& datasetPath);
};
