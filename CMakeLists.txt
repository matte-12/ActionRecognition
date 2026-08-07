cmake_minimum_required(VERSION 3.10)

project(ActionRecognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(ActionRecognition
    main.cpp
    src/segmentation/Segmentation.cpp
    src/segmentation/MotionMask.cpp
    src/segmentation/GrabCutSegmentation.cpp
    src/segmentation/BoundingBox.cpp
    src/segmentation/ShadowRemoval.cpp
    src/segmentation/MediaPipeFallback.cpp
    src/segmentation/VisualizationTest.cpp
    src/SavedSegmentationLoader.cpp
    src/DatasetLoader.cpp
    src/Visualizer.cpp
    src/FeatureExtractor.cpp
    src/Classifier.cpp
    src/Metrics.cpp


)

add_executable(annotate_gt
    tools/annotate_gt.cpp
)

target_link_libraries(annotate_gt
    ${OpenCV_LIBS}
)

target_link_libraries(ActionRecognition
    ${OpenCV_LIBS}
)
