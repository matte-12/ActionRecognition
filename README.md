Place the dataset inside:

data/sequences/

NB: il dataset andrebbe messo in locale, siccome però non è ancora stato caricato dal prof. allora ne ho fatto un mio. 
    Tolgo da .gitignore il dataset, cosi carico le cartelle.
//---------------------------------------------------------
Codici che uso per runnare:

> motion_mask_stats.txt
rm -rf build
mkdir build
cd build

cmake ..
make

cd ..
./build/ActionRecognition data/sequences results

//----------------------------------------------------------
Codice da usare per killare mediapipe:

pkill -9 -f ActionRecognition
