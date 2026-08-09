#include "Metrics.hpp"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;


// ------------------------------------------------------------


static double computePrecision(
    const int confusionMatrix[7][7],
    int classId)
{
    int truePositive = confusionMatrix[classId][classId];

    int falsePositive = 0;

    for (int trueLabel = 1; trueLabel <= 6; trueLabel++)
    {
        if (trueLabel != classId)
        {
            falsePositive += confusionMatrix[trueLabel][classId];
        }
    }

    int denominator = truePositive + falsePositive;

    if (denominator == 0)
        return 0.0;

    return static_cast<double>(truePositive) /
           static_cast<double>(denominator);
}

// ------------------------------------------------------------


static double computeRecall(
    const int confusionMatrix[7][7],
    int classId)
{
    int truePositive = confusionMatrix[classId][classId];

    int falseNegative = 0;

    for (int predictedLabel = 1; predictedLabel <= 6; predictedLabel++)
    {
        if (predictedLabel != classId)
        {
            falseNegative += confusionMatrix[classId][predictedLabel];
        }
    }

    int denominator = truePositive + falseNegative;

    if (denominator == 0)
        return 0.0;

    return static_cast<double>(truePositive) /
           static_cast<double>(denominator);
}

// ------------------------------------------------------------


static double computeF1Score(
    double precision,
    double recall)
{
    if (precision + recall == 0.0)
        return 0.0;

    return 2.0 * precision * recall / (precision + recall);
}

// ------------------------------------------------------------

static void printF1Scores(
    const int confusionMatrix[7][7])
{
    std::cout << std::endl;
    std::cout << "F1-score per class" << std::endl;
    std::cout << "Class | Precision | Recall | F1-score" << std::endl;

    for (int classId = 1; classId <= 6; classId++)
    {
        double precision =
            computePrecision(confusionMatrix, classId);

        double recall =
            computeRecall(confusionMatrix, classId);

        double f1 =
            computeF1Score(precision, recall);

        std::cout << std::setw(5) << classId << " | "
                  << std::setw(9) << precision << " | "
                  << std::setw(6) << recall << " | "
                  << std::setw(8) << f1
                  << std::endl;
    }
}

// ------------------------------------------------------------


static cv::Rect2d convertRectToRect2d(const cv::Rect& rect)
{
    return cv::Rect2d(
        static_cast<double>(rect.x),
        static_cast<double>(rect.y),
        static_cast<double>(rect.width),
        static_cast<double>(rect.height)
    );
}

// ------------------------------------------------------------


static double computeIntersectionArea(
    const cv::Rect2d& boxA,
    const cv::Rect2d& boxB)
{
    double xLeft =
        std::max(boxA.x, boxB.x);

    double yTop =
        std::max(boxA.y, boxB.y);

    double xRight =
        std::min(boxA.x + boxA.width, boxB.x + boxB.width);

    double yBottom =
        std::min(boxA.y + boxA.height, boxB.y + boxB.height);

    double intersectionWidth =
        std::max(0.0, xRight - xLeft);

    double intersectionHeight =
        std::max(0.0, yBottom - yTop);

    return intersectionWidth * intersectionHeight;
}

// ------------------------------------------------------------


static double computeIoU(
    const cv::Rect& predictedBox,
    const cv::Rect2d& groundTruthBox)
{
    if (predictedBox.area() <= 0 || groundTruthBox.area() <= 0.0)
        return 0.0;

    cv::Rect2d predictedBoxDouble = convertRectToRect2d(predictedBox);

    double intersectionArea = 
        computeIntersectionArea(
            predictedBoxDouble,
            groundTruthBox
        );

    double unionArea = predictedBoxDouble.area() + groundTruthBox.area() - intersectionArea;

    if (unionArea <= 0.0)
        return 0.0;

    return intersectionArea / unionArea;
}

// ------------------------------------------------------------


static bool isValidLabel(int label)
{
    return label >= 1 && label <= 6;
}

// ------------------------------------------------------------


static void createParentDirectoryIfNeeded(
    const std::string& outputPath)
{
    fs::path path(outputPath);
    fs::path parentPath = path.parent_path();

    if (!parentPath.empty())
    {
        fs::create_directories(parentPath);
    }
}

// ------------------------------------------------------------


static void printConfusionMatrix(
    const int confusionMatrix[7][7])
{
    std::cout << std::endl;
    std::cout << "Confusion Matrix" << std::endl;
    std::cout << "Rows = true label, Columns = predicted label" << std::endl;

    std::cout << "      ";

    for (int predicted = 1; predicted <= 6; predicted++)
    {
        std::cout << std::setw(4) << predicted;
    }

    std::cout << std::endl;

    for (int trueLabel = 1; trueLabel <= 6; trueLabel++)
    {
        std::cout << std::setw(4) << trueLabel << " ";

        for (int predicted = 1; predicted <= 6; predicted++)
        {
            std::cout << std::setw(4)
                      << confusionMatrix[trueLabel][predicted];
        }

        std::cout << std::endl;
    }
}

// ------------------------------------------------------------

    static void writeF1Scores(
        std::ofstream& file,
        const int confusionMatrix[7][7])
    {
        file << std::endl;
        file << "F1-score per class" << std::endl;
        file << "Class | Precision | Recall | F1-score" << std::endl;

        for (int classId = 1; classId <= 6; classId++)
        {
            double precision =
                computePrecision(confusionMatrix, classId);

            double recall =
                computeRecall(confusionMatrix, classId);

            double f1 =
                computeF1Score(precision, recall);

            file << std::setw(5) << classId << " | "
                << std::setw(9) << precision << " | "
                << std::setw(6) << recall << " | "
                << std::setw(8) << f1
                << std::endl;
        }
    }

// ------------------------------------------------------------

    static void writeConfusionMatrix(
        std::ofstream& file,
        const int confusionMatrix[7][7])
    {
        file << std::endl;
        file << "Confusion Matrix" << std::endl;
        file << "Rows = true label, Columns = predicted label" << std::endl;

        file << "      ";

        for (int predicted = 1; predicted <= 6; predicted++)
        {
            file << std::setw(4) << predicted;
        }

        file << std::endl;

        for (int trueLabel = 1; trueLabel <= 6; trueLabel++)
        {
            file << std::setw(4) << trueLabel << " ";

            for (int predicted = 1; predicted <= 6; predicted++)
            {
                file << std::setw(4)
                    << confusionMatrix[trueLabel][predicted];
            }

            file << std::endl;
        }
    }


// ------------------------------------------------------------


static void writeCSVHeader(std::ofstream& file)
{
    file << "sequence,true_label,predicted_label,iou_frame20\n";
}

// ------------------------------------------------------------
// Metrics methods
// ------------------------------------------------------------

void Metrics::addResult(
    const SequenceData& sequence,
    int predictedLabel)
{
    Result result;

    result.sequenceName = sequence.name;
    result.trueLabel = sequence.trueLabel;
    result.predictedLabel = predictedLabel;
    result.iouFrame20 = 0.0;

    if (sequence.hasGT && sequence.BBoxes.size() > 19)
    {
        result.iouFrame20 =
            computeIoU(
                sequence.BBoxes[19],
                sequence.gtBBox20
            );
         result.hasIoU = true;
    }

    results.push_back(result);
}

// ------------------------------------------------------------

void Metrics::saveSummaryCSV(
    const std::string& outputPath) const
{
    createParentDirectoryIfNeeded(outputPath);

    std::ofstream file(outputPath);

    if (!file.is_open())
    {
        std::cerr << "Cannot open metrics summary CSV: "
                  << outputPath << std::endl;
        return;
    }

    if (results.empty())
    {
        file << "metric,value\n";
        file << "total_results,0\n";
        return;
    }

    int correct = 0;
    int validLabels = 0;

    double meanIoU = 0.0;
    int validIoU = 0;

    int confusionMatrix[7][7] = {};

    for (const Result& result : results)
    {
        if (isValidLabel(result.trueLabel) &&
            isValidLabel(result.predictedLabel))
        {
            validLabels++;

            if (result.trueLabel == result.predictedLabel)
            {
                correct++;
            }

            confusionMatrix[result.trueLabel][result.predictedLabel]++;
        }

        if (result.iouFrame20 > 0.0)
        {
            meanIoU += result.iouFrame20;
            validIoU++;
        }
    }

    double accuracy = 0.0;

    if (validLabels > 0)
    {
        accuracy = static_cast<double>(correct) / static_cast<double>(validLabels);
    }

    if (validIoU > 0)
    {
        meanIoU /= static_cast<double>(validIoU);
    }

    file << "metric,value\n";
    file << "total_results," << results.size() << "\n";
    file << "valid_labels," << validLabels << "\n";
    file << "correct_predictions," << correct << "\n";
    file << "global_accuracy," << accuracy << "\n";
    file << "mean_iou_frame20," << meanIoU << "\n";
    file << "\n";

    file << "class,precision,recall,f1_score\n";

    for (int classId = 1; classId <= 6; classId++)
    {
        double precision = computePrecision(confusionMatrix, classId);

        double recall = computeRecall(confusionMatrix, classId);

        double f1 = computeF1Score(precision, recall);

        file << classId << ","
             << precision << ","
             << recall << ","
             << f1 << "\n";
    }

    file << "\n";

    file << "true_label,predicted_1,predicted_2,predicted_3,predicted_4,predicted_5,predicted_6\n";

    for (int trueLabel = 1; trueLabel <= 6; trueLabel++)
    {
        file << trueLabel;

        for (int predictedLabel = 1; predictedLabel <= 6; predictedLabel++)
        {
            file << "," << confusionMatrix[trueLabel][predictedLabel];
        }

        file << "\n";
    }

    file.close();
}
// ------------------------------------------------------------

    void Metrics::printSummary() const
    {
        if (results.empty())
        {
            std::cout << "No metrics available." << std::endl;
            return;
        }

        int correct = 0;
        int validLabels = 0;

        double meanIoU = 0.0;
        int validIoU = 0;

        int confusionMatrix[7][7] = {};

        for (const Result& result : results)
        {
            if (isValidLabel(result.trueLabel) &&
                isValidLabel(result.predictedLabel))
            {
                validLabels++;

                if (result.trueLabel == result.predictedLabel)
                {
                    correct++;
                }

                confusionMatrix[result.trueLabel][result.predictedLabel]++;
            }

            if (result.hasIoU)
            {
                meanIoU += result.iouFrame20;
                validIoU++;
            }
        }

        double accuracy = 0.0;

        if (validLabels > 0)
        {
            accuracy = static_cast<double>(correct) /static_cast<double>(validLabels);
        }

        if (validIoU > 0)
        {
            meanIoU /= static_cast<double>(validIoU);
        }

        std::cout << "========== METRICS ==========" << std::endl;
        std::cout << "Total results: " << results.size() << std::endl;
        std::cout << "Valid labels: " << validLabels << std::endl;
        std::cout << "Correct predictions: " << correct << std::endl;
        std::cout << "Global Accuracy: " << accuracy << std::endl;
        std::cout << "Mean IoU frame 20: " << meanIoU << std::endl;

        printConfusionMatrix(confusionMatrix);

        printF1Scores(confusionMatrix);

        std::cout << "=============================" << std::endl;
    }

    // ------------------------------------------------------------


    void Metrics::saveResultsCSV(
        const std::string& outputPath) const
    {
        createParentDirectoryIfNeeded(outputPath);

        std::ofstream file(outputPath);

        if (!file.is_open())
        {
            std::cerr << "Cannot open results CSV: "
                    << outputPath << std::endl;
            return;
        }

        writeCSVHeader(file);

        for (const Result& result : results)
        {
            file << result.sequenceName << ","
                << result.trueLabel << ","
                << result.predictedLabel << ","
                << result.iouFrame20 << "\n";
        }

        file.close();
    }