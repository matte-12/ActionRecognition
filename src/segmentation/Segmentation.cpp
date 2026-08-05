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


// NB HA ANCORA PROBLEMI CON OMBRE E CAMBI DI LUCE

// funzione public che estrae le BBox finali 

void Segmentation::extractBBox( SequenceData& sequence) {



    std::vector<cv::Mat> filteredImgs; //vettore con immagini che vengo modificate e filtrate

    // applico mog2 per ricavare foreground
    
    filteredImgs = (filterMOG2(sequence.frames)) ; 


    //applico morphology
    // NB: HO NOTATO CHE CON KERNEL (3,3) PRENDE ANCHE HANDCLAPPING, CON KERNEL (5,5) NON CI RIESCE
    // IN QUESTO MODO NON USA FALLBACK, PER0' E' UN PO' PIU' IMPRECISO CON I CAMBI DI ILLUMINAZIONE
    // NB: CAMBIO LA THRESHOLD PER EFFETTI SULL'ILLUMINAZIONE

    filteredImgs = (filterMorphology(filteredImgs));


    //  showMasks(filteredImgs); 

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

    // Salvo nella struct sequences le final mask

    sequence.finalMasks = finalImgsMask;


    //mostra se funziona maschera
    // showMasks(finalImgsMask);  

    // ritorna vettore di Rect

    std::vector<cv::Rect> bboxes = findBBox(finalImgsMask);

    // Salvo nella struct sequences le BBoxes
    
    sequence.BBoxes = bboxes;

    // SMOOTH BBOX
    // lo smoothing non risponde benissimo quando i soggetti corrono

    // bboxes = smoothBBoxes(bboxes, 0.50, 3);

    // mostra bbox
    //  showBBoxes(sequence.frames, bboxes);

}
//---------------------------------------------------------------------------------------
