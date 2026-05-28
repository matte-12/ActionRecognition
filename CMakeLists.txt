cmake_minimum_required(VERSION 3.10)

project(ActionRecognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(ActionRecognition
    main.cpp
)

target_link_libraries(ActionRecognition
    ${OpenCV_LIBS}
)
