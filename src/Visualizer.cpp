#include "Visualizer.hpp"


//SALVA LE MASK FINALI NELLA STESSA CARTELLA DELLE SEQUENZE (SEMPRE DENTRO RESULTS)

void Visualizer::saveMasks(
    const std::string& sequenceName,
    const std::vector<cv::Mat>& maschera ,
    const std::string& outputFolder,
    const std::string lastFolderName)
{
    namespace fs = std::filesystem;

    fs::path sequenceOutputPath = fs::path(outputFolder) / sequenceName / lastFolderName;

    if (!fs::exists(sequenceOutputPath))
    {
        fs::create_directories(sequenceOutputPath);
    }

    for (size_t i = 0; i < maschera.size(); i++)
    {
        const cv::Mat& mask = maschera[i];

        if (mask.empty())
            continue;

        cv::Mat maskToSave;

        if (mask.channels() == 1)
        {
            maskToSave = mask.clone();
        }
        else
        {
            cv::cvtColor(mask, maskToSave, cv::COLOR_BGR2GRAY);
        }

        cv::threshold(
            maskToSave,
            maskToSave,
            127,
            255,
            cv::THRESH_BINARY
        );

        std::ostringstream filename;

        filename << "mask_result"
                 << std::setw(3)
                 << std::setfill('0')
                 << i + 1
                 << ".jpg";

        fs::path outputPath = sequenceOutputPath / filename.str();

        bool saved = cv::imwrite(outputPath.string(), maskToSave);

        if (!saved)
        {
            std::cerr << "Errore salvataggio mask: "
                      << outputPath.string()
                      << std::endl;
        }
    }
}

//-------------------------------------------------------------------------------------------

// SALVA LE IMMAGINI NELLA CARTELLA results/

void Visualizer:: salvataggio (cv:: Mat& img, fs::path& sequenceOutputPath, int counter){

    if (!fs::exists(sequenceOutputPath))
    {
        fs::create_directories(sequenceOutputPath);
    }

      std::ostringstream filename;

      filename << "frame_result"
                 << std::setw(3)
                 << std::setfill('0')
                 << counter
                 << ".jpg";

        fs::path outputPath = sequenceOutputPath / filename.str();

        bool saved = cv::imwrite(outputPath.string(), img);

        if (!saved)
        {
            std::cerr << "Errore salvataggio immagine: "
                      << outputPath.string()
                      << std::endl;
        }

}

//-------------------------------------------------------------------------------------------

// UNISCO FRAME BBOX CON LABEL

cv::Mat Visualizer :: unione (cv::Mat& frame, std::string& label, cv::Rect& bbox){

    if (frame.empty())
            return frame;

        cv::Mat display;

// aggiungi bbox al frame

        if (frame.channels() == 1)
            cv::cvtColor(frame, display, cv::COLOR_GRAY2BGR);
        else
            display = frame.clone();

        if (bbox.area() > 0)
        {
            cv::rectangle(
                display,
                bbox,
                cv::Scalar(0, 255, 0),
                2
            );
        }

// aggiungi label

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(
            label,
            cv::FONT_HERSHEY_DUPLEX,
            0.5,
            1,
            &baseline
        );

        // Coordinate del testo
        int x = (frame.cols - textSize.width) / 2;
        int y = frame.rows - 20;   // 20 pixel sopra il bordo inferiore

        cv::putText(
            display,
            label,
            cv::Point(x, y),
            cv::FONT_HERSHEY_DUPLEX,
            0.5,
            cv::Scalar(0, 255, 0),
            1
        );

    return display;

}

//-------------------------------------------------------------------------------------------
// 1= Boxing 2= handclapping 3= handwaving 4= jogging 5= running 6= walking
std::string Visualizer :: conversione(int predictedLabel){

    switch(predictedLabel){
         case 1: return "boxing";
        case 2: return "handclapping";
        case 3: return "handwaving";
        case 4: return "jogging";
        case 5: return "running";
        case 6: return "walking";
        default: return "unknown";
    }

    return "unknown";
       
}

//-------------------------------------------------------------------------------------------

//METODO PUBLIC PRINCIPALE

void Visualizer::saveSequenceOutput(
    const SequenceData& sequenza,
    int predictedLabel,
    const std::string& outputFolder
    ) 
{

// converto la label da int a string

    std::string label = conversione(predictedLabel);

// unisco frame, bbox e label. Ritorno una unica Mat img

    std::vector <cv::Mat> imgs; 

 // Cartella finale: results/nome_sequenza/
    fs::path sequenceOutputPath = fs::path(outputFolder) / sequenza.name / "output";

    for(size_t i = 0; i< sequenza.frames.size(); i++){

        cv:: Mat frame = sequenza.frames[i];
        cv:: Rect bbox = sequenza.BBoxes[i];
        cv:: Mat img = unione(frame, label, bbox);

        imgs.push_back(img);

        //salva le immagini 
        salvataggio(img, sequenceOutputPath, i);
    }

// salva le fianl masks

    saveMasks(sequenza.name, sequenza.finalMasks, "results", "masks");
    saveMasks(sequenza.name, sequenza.motionMasks, "results", "motion_masks");
}

//-------------------------------------------------------------------------------------------