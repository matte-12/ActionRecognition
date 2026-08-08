#include "featExtraction/FeatureExtractor.hpp"
#include "featExtraction/MotionStats.hpp"
#include "featExtraction/OpticalFlow.hpp"

#include <opencv2/opencv.hpp>

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

//------------------------------------------------------------

static double pointDistance(
    const cv::Point2d& a,
    const cv::Point2d& b)
{
    double dx = b.x - a.x;
    double dy = b.y - a.y;

    return std::sqrt(dx * dx + dy * dy);
}



//------------------------------------------------------------

static std::vector<double> computeWidths(
    const std::vector<cv::Rect>& boxes)
{
    std::vector<double> widths;

    for (const cv::Rect& box : boxes)
    {
        if (box.width > 0)
            widths.push_back(static_cast<double>(box.width));
    }

    return widths;
}

//------------------------------------------------------------

static std::vector<double> computeHeights(
    const std::vector<cv::Rect>& boxes)
{
    std::vector<double> heights;

    for (const cv::Rect& box : boxes)
    {
        if (box.height > 0)
            heights.push_back(static_cast<double>(box.height));
    }

    return heights;
}

//------------------------------------------------------------
// tiene solo le BBox valide. Evita BBox vuote come schermo nero

static std::vector<cv::Rect> getValidBoxes(
    const std::vector<cv::Rect>& boxes)
{
    std::vector<cv::Rect> validBoxes;

    for (const cv::Rect& box : boxes)
    {
        if (box.width > 0 && box.height > 0 && box.area() > 0)
        {
            validBoxes.push_back(box);
        }
    }

    return validBoxes;
}

// ------------------------------------------------------------
// trova area delle BBox

static std::vector<double> computeAreas(
    const std::vector<cv::Rect>& boxes)
{
    std::vector<double> areas;

    for (const cv::Rect& box : boxes)
    {
        areas.push_back(static_cast<double>(box.area()));
    }

    return areas;
}

// ------------------------------------------------------------
// ratio larghezza su altezza. Aumenta se body apre le braccia

static std::vector<double> computeAspectRatios(
    const std::vector<double> width,
    const std::vector<double> height
    )
{
    std::vector<double> aspectRatios;

    for (size_t i = 0; i < width.size(); ++i)
    {

        double aspectRatio = width[i] / height[i];

        aspectRatios.push_back(aspectRatio);
    }

    return aspectRatios;
}

//-----------------------------------------------------------
//calcola singolo centro
static cv::Point2d getSingleCenter(
    const cv::Rect& box)
{
    return cv::Point2d(
        box.x + box.width / 2.0,
        box.y + box.height / 2.0
    );
}

//-----------------------------------------------------------
// calcola centro della BBox (larghezza, altezza)

static std::vector<cv::Point2d> computeCenters(
    const std::vector<cv::Rect>& boxes)
{
    std::vector<cv::Point2d> centers;

    for (const cv::Rect& box : boxes)
    {
        cv::Point2d center(
            box.x + box.width / 2.0,
            box.y + box.height / 2.0
        );

        centers.push_back(center);
    }

    return centers;
}

// ------------------------------------------------------------
// calcola area media 

static double computeMean( const std::vector<double>& values )
{
    if (values.empty())
        return 0.0;

    double sum = 0.0;

    for (double value : values)
    {
        sum += value;
    }

    return sum / static_cast<double>(values.size());
}

// ------------------------------------------------------------
// calcola std deviation area

static double computeStd(
    const std::vector<double>& values,
    double mean)
{
    if (values.empty())
        return 0.0;

    double sum = 0.0;

    for (double value : values)
    {
        double diff = value - mean;
        sum += diff * diff;
    }

    return std::sqrt(sum / static_cast<double>(values.size()));
}

// ------------------------------------------------------------
// da rivedere
static double computeTotalDisplacementX(
    const std::vector<cv::Point2d>& centers)
{
    if (centers.size() < 2)
        return 0.0;

    return centers.back().x - centers.front().x;
}

// ------------------------------------------------------------

static double computeTotalDisplacementY(
    const std::vector<cv::Point2d>& centers)
{
    if (centers.size() < 2)
        return 0.0;

    return centers.back().y - centers.front().y;
}

// ------------------------------------------------------------
// total displacement distanza euclidea

static double computeTotalDisplacement(
    const std::vector<cv::Point2d>& centers)
{
    if (centers.size() < 2)
        return 0.0;

    double dx = centers.back().x - centers.front().x;
    double dy = centers.back().y - centers.front().y;

    return std::sqrt(dx * dx + dy * dy);
}

// ------------------------------------------------------------

static double computeMeanVelocity(
    const std::vector<cv::Point2d>& centers)
{
    if (centers.size() < 2)
        return 0.0;

    double totalPath = 0.0;

    for (size_t i = 1; i < centers.size(); i++)
    {
        double dx = centers[i].x - centers[i - 1].x;
        double dy = centers[i].y - centers[i - 1].y;

        totalPath += std::sqrt(dx * dx + dy * dy);
    }

    return totalPath / static_cast<double>(centers.size() - 1);
}

// ------------------------------------------------------------
// conta la media dei pixel bianchi guardando tutte le mask
// non mi serve
static double computeMotionEnergy(
    const std::vector<cv::Mat>& masks)
{
    double totalEnergy = 0.0;
    int validMasks = 0;

    for (const cv::Mat& mask : masks)
    {
        if (mask.empty())
            continue;

        double totalPixels = static_cast<double>(mask.total());

        if (totalPixels <= 0.0)
            continue;

        double foregroundPixels = static_cast<double>(cv::countNonZero(mask));

        totalEnergy += foregroundPixels / totalPixels;
        validMasks++;
    }

    if (validMasks == 0)
        return 0.0;

    return totalEnergy / static_cast<double>(validMasks);
}

//------------------------------------------------------------

static double computePathLength(
    const std::vector<cv::Rect>& boxes)
{
    double totalPath = 0.0;

    for (size_t i = 1; i < boxes.size(); i++)
    {
        if (boxes[i - 1].area() <= 0 ||
            boxes[i].area() <= 0)
        {
            continue;
        }

        cv::Point2d previousCenter =
            getSingleCenter(boxes[i - 1]);

        cv::Point2d currentCenter =
            getSingleCenter(boxes[i]);

        totalPath +=
            pointDistance(previousCenter, currentCenter);
    }

    return totalPath;
}


// ------------------------------------------------------------
// ------------------------------------------------------------
// Main feature extraction 


SequenceFeatures FeatureExtractor::extract(const SequenceData& sequence)
{
    SequenceFeatures features;

    features.sequenceName = sequence.name;
    features.trueLabel = sequence.trueLabel;

    std::vector<cv::Rect> validBoxes = getValidBoxes(sequence.BBoxes);

    std::vector<double> areas = computeAreas(validBoxes);

    std::vector<double> widths = computeWidths(validBoxes);
    std::vector<double> heights = computeHeights(validBoxes);

    //width / height
    std::vector<double> aspectRatios = computeAspectRatios(widths, heights);

    // calcola i centri delle 
    std::vector<cv::Point2d> centers = computeCenters(validBoxes);

    // calcolo bbox features
    features.meanArea = computeMean(areas);

    features.areaVariation = computeStd(areas, features.meanArea);

    features.meanAspectRatio = computeMean(aspectRatios);

    features.aspectRatioVariation = computeStd(aspectRatios, features.meanAspectRatio);
    
    features.meanBoxWidth = computeMean(widths);

    features.meanBoxHeight = computeMean(heights);

    features.widthVariation = computeStd( widths, features.meanBoxWidth );

    if (features.meanBoxWidth > 0.0)
    {
        features.widthVariationNorm =
            features.widthVariation /
            features.meanBoxWidth;
    }

    // calcolo lo spostamento

    features.totalDisplacementX = computeTotalDisplacementX(centers);

    features.totalDisplacementY = computeTotalDisplacementY(centers);

    features.totalDisplacement = computeTotalDisplacement(centers);

    features.meanVelocity = computeMeanVelocity(centers);
    // motionRnergy su finalMask
    features.motionEnergyFinalMask = computeMotionEnergy(sequence.finalMasks);

    features.pathLength = computePathLength(sequence.BBoxes);

    if (features.meanBoxHeight > 0.0)
    {
        features.displacementNorm =
            features.totalDisplacement /
            features.meanBoxHeight;

        features.pathLengthNorm =
            features.pathLength /
            features.meanBoxHeight;

        features.velocityNorm =
            features.meanVelocity /
            features.meanBoxHeight;
    }

    if (features.pathLength > 0.0)
    {
        features.trajectoryEfficiency =
            features.totalDisplacement /
            features.pathLength;
    }

    // calcolo motion del corpo dividendo la bbox in 9 blocchi

    MotionStats motion = computeBodyMotion(sequence.motionMasks, sequence.BBoxes);
    // motion enegy solo su motionMask
    features.motionEnergySeed = motion.motionEnergy;

    features.upperMotion = motion.upper;

    features.middleMotion = motion.middle;

    features.lowerMotion = motion.lower;

    features.leftMotion = motion.left;

    features.rightMotion = motion.right;

    features.leftRightDifference = std::abs(motion.left - motion.right);

    features.centerMotionRatio = motion.centerRatio;

    //optical flow

    OpticalFlow flow = computeOpticalFlow(sequence);

    features.meanFlowMagnitude = flow.meanFlowMagnitude;

    features.maxFlowMagnitude = flow.maxFlowMagnitude;

    features.upperHorizontalFlow = flow.upperHorizontalFlow;

    features.upperVerticalFlow = flow.upperVerticalFlow;

    features.flowConvergence = flow.flowConvergence;

    features.lowerFlowMagnitude = flow.lowerFlowMagnitude;

    features.leftX = flow.leftX;

    features.rightX = flow.rightX;

    features.leftAbsX = flow.leftAbsX;

    features.rightAbsX = flow.rightAbsX;

    features.horizontalSymmetry = flow.horizontalSymmetry;

    features.strideFrequency = flow.strideFrequency;

    return features;
}

// ------------------------------------------------------------
// funzione pubblica per i log

void FeatureExtractor::logFeatures(
    const SequenceFeatures& features,
    const std::string& logPath)
{
    fs::path path(logPath);
    fs::path parentPath = path.parent_path();

    if (!parentPath.empty())
    {
        fs::create_directories(parentPath);
    }

    std::ofstream file(logPath, std::ios::app);

    bool fileAlreadyExists = fs::exists(path) && fs::file_size(path) > 0;

    if (!file.is_open())
    {
        std::cerr << "Cannot open feature log: "
                  << logPath << std::endl;
        return;
    }

    if (!fileAlreadyExists)
    {
        file << "sequence,true_label,"
        << "mean_area,area_variation,"
        << "mean_aspect_ratio,aspect_ratio_variation,"
        << "mean_box_width,mean_box_height,"
        << "width_variation,width_variation_norm,"
        << "total_displacement_x,total_displacement_y,total_displacement,"
        << "displacement_norm,"
        << "path_length,path_length_norm,"
        << "mean_velocity,velocity_norm,"
        << "motion_energy_finalMask,"
        << "trajectory_efficiency,"
        << "motion_energy_seed,"
        << "upper_motion,middle_motion,lower_motion,"
        << "left_motion,right_motion,left_right_difference,"
        << "center_motion_ratio"
        << ",mean_flow_magnitude"
        << ",max_flow_magnitude"
        << ",upper_horizontal_flow"
        << ",upper_vertical_flow"
        << ",flow_convergence"
        << ",lower_flow_magnitude"
        << ",left_x"
        << ",right_x"
        << ",left_abs_x"
        << ",right_abs_x"
        << ",horizontal_symmetry"
        << ",stride_frequency"
        << "\n";
    }

    file << features.sequenceName << ","
        << features.trueLabel << ","

        << features.meanArea << ","
        << features.areaVariation << ","

        << features.meanAspectRatio << ","
        << features.aspectRatioVariation << ","

        << features.meanBoxWidth << ","
        << features.meanBoxHeight << ","

        << features.widthVariation << ","
        << features.widthVariationNorm << ","

        << features.totalDisplacementX << ","
        << features.totalDisplacementY << ","
        << features.totalDisplacement << ","

        << features.displacementNorm << ","

        << features.pathLength << ","
        << features.pathLengthNorm << ","

        << features.meanVelocity << ","
        << features.velocityNorm << ","

        << features.motionEnergyFinalMask << ","
        << features.trajectoryEfficiency << ","

        << features.motionEnergySeed << ","

        << features.upperMotion << ","
        << features.middleMotion << ","
        << features.lowerMotion << ","

        << features.leftMotion << ","
        << features.rightMotion << ","
        << features.leftRightDifference << ","

        << features.centerMotionRatio << ","
        << features.meanFlowMagnitude
        << ","
        << features.maxFlowMagnitude
        << ","
        << features.upperHorizontalFlow
        << ","
        << features.upperVerticalFlow
        << ","
        << features.flowConvergence
        << ","
        << features.lowerFlowMagnitude
        << ","
        << features.leftX
        << ","
        << features.rightX
        << ","
        << features.leftAbsX
        << ","
        << features.rightAbsX
        << ","
        << features.horizontalSymmetry
        << ","
        << features.strideFrequency
        << "\n";

    file.close();

}