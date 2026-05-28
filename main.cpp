#include <iostream>

int main(int argc, char** argv) {
    std::cout << "Action Recognition project started!" << std::endl;

    if (argc >= 3) {
        std::cout << "Dataset path: " << argv[1] << std::endl;
        std::cout << "Results path: " << argv[2] << std::endl;
    } else {
        std::cout << "Usage: ./ActionRecognition <dataset_path> <results_path>" << std::endl;
    }

    return 0;
}
