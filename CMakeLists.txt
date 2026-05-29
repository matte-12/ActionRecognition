cmake_minimum_required(VERSION 3.10)

project(ActionRecognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(ActionRecognition
    main.cpp

    src/DatasetLoader.cpp
    src/Segmentation.cpp
    src/Visualizer.cpp
    src/FeatureExtractor.cpp
    src/Classifier.cpp
    src/Metrics.cpp
)

target_link_libraries(ActionRecognition
    ${OpenCV_LIBS}
)
