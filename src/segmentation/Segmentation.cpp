#include "segmentation/Segmentation.hpp"
#include <fstream>
#include <cstdlib>
#include <cmath>
#include "segmentation/MediaPipeFallback.hpp"
#include "segmentation/MotionMask.hpp"
#include "segmentation/ShadowRemoval.hpp"
#include "segmentation/GrabCutSegmentation.hpp"
#include "segmentation/BoundingBox.hpp"
#include "segmentation/VisualizationTest.hpp"


static bool removeTinyMasks(
     std::vector<cv::Mat>& masks,
    int minNonZero, std::string name)
{

    int count = 0;

    for (size_t i = 0; i < masks.size(); i++)
    {
        int nonZero = cv::countNonZero(masks[i]);

        if (nonZero > 0 && nonZero < minNonZero )
        {
            masks[i] = cv::Mat::zeros(masks[i].size(), CV_8UC1);
            count++;
        }
    
    }

    std::cout << "[removeTinyMasks] removed tiny masks from" + name + ": "
            << count << std::endl;

    return count > 0;
}

//--------------------------------------------------------------------------------------

// NB HA ANCORA PROBLEMI CON OMBRE E CAMBI DI LUCE
// funzione public che estrae le BBox finali 

void Segmentation::extractBBox( SequenceData& sequence) {

    //vettore con immagini che vengo modificate e filtrate
    std::vector<cv::Mat> filteredImgs; 

    // applico mog2 per ricavare foreground
    filteredImgs = (filterMOG2(sequence.frames)) ; 

    //applico morphology
    // NB: HO NOTATO CHE CON KERNEL (3,3) PRENDE ANCHE HANDCLAPPING, CON KERNEL (5,5) NON CI RIESCE
    // IN QUESTO MODO NON USA FALLBACK, PER0' E' UN PO' PIU' IMPRECISO CON I CAMBI DI ILLUMINAZIONE
    // NB: CAMBIO LA THRESHOLD PER EFFETTI SULL'ILLUMINAZIONE

    filteredImgs = (filterMorphology(filteredImgs));

    // Salvo la motion mask (seeds) in sequence.motionMasks (utile per feature extraction)

    sequence.motionMasks = filteredImgs;

    //  Decido se la sequenza è da MediaPipe

    bool useMediaPipeForSequence = shouldUseMediaPipeForSequence(filteredImgs, 0.80);

    // Se NON è una sequenza da MediaPipe,
    // allora riempio temporalmente i seed vuoti. TRICK!

    if (!useMediaPipeForSequence)
    {
        filteredImgs = fillEmptySeedsTemporal(filteredImgs);
    }       

    // Cerco maschera finale: applico GrabCut, se maschera ancora nera uso GrabcutMediaPipe
    std::vector<cv::Mat> finalImgsMask; 

    finalImgsMask = getFinalMask(
        sequence.frames,
        filteredImgs,
        sequence.name,
        useMediaPipeForSequence
        );

    // // pulisco la mask dalle ombre

        for (cv::Mat& mask : finalImgsMask)
    {
        mask = removeShadow(mask);
    }
    
    // //applico ancora un fill temporale perchè alcune images sono diventate nere
    // // causa vincoli forti si removeShadow()
    // filteredImgs = fillEmptySeedsTemporal(filteredImgs);

    // se grabcut non è riuscito a ricostruire l'img da piccoli elementi (area<200)
    // nota che soglia molto alta perche dal log dopo grabcut vanno quesi tutti a 1000 nonZero, tranne i casi in questione
    // allora faccio fill temporale
    bool flag = removeTinyMasks(finalImgsMask, 120, sequence.name);

    if(flag)
        filteredImgs = fillEmptySeedsTemporal(filteredImgs);

    // // Salvo nella struct sequences la final mask
    sequence.finalMasks = finalImgsMask;

    //mostra se funziona maschera
    // showMasks(finalImgsMask);  

    // ritorna vettore di Rect
    std::vector<cv::Rect> bboxes = findBBox(finalImgsMask);

    // Salvo nella struct sequences le BBoxes
    sequence.BBoxes = bboxes;

    // mostra bbox
    //  showBBoxes(sequence.frames, bboxes);

}
//---------------------------------------------------------------------------------------
