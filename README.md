Dataset inside folder:

local_sequences/data 

Inside folder /results we have:

classification_results_loso.csv
metrics_summary_loso.csv
local_sequences/

Inside results/local_sequences/.../ folder:

masks/ --> final masks
motion_masks/  --> evaluated from mog2
output/  --> final BBox and labels

From the project root, to make build and run:

```bash
rm -rf build
cmake -S . -B build
cmake --build build

./build/ActionRecognition local_sequences results

```md
Usage:

```bash
./build/ActionRecognition <dataset_path> <results_path>


'''md
Code to eventually kill mediapipe execution:
'''bash
 pkill -9 -f ActionRecognition
