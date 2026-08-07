#pragma once

#include "Types.hpp"

#include <string>

bool loadSavedSegmentation(
    SequenceData& sequence,
    const std::string& resultsRoot);