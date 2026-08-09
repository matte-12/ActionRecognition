#pragma once

#include "Types.hpp"

#include <string>
#include <vector>

class Metrics
{
public:
    void addResult(
        const SequenceData& sequence,
        int predictedLabel);

    void printSummary() const;

    void saveResultsCSV(const std::string& outputPath) const;

    void saveSummaryCSV(const std::string& outputPath) const;

private:
    struct Result
    {
        std::string sequenceName;
        int trueLabel = -1;
        int predictedLabel = -1;
        double iouFrame20 = 0.0;
        bool hasIoU = false;
    };

    std::vector<Result> results;
};