Place the dataset inside:

data/sequences/

Codici che uso per runnare:

> motion_mask_stats.txt
rm -rf build
mkdir build
cd build

cmake ..
make

cd ..
./build/ActionRecognition data/sequences results


Codice da usare per killare mediapipe:

pkill -9 -f ActionRecognition
