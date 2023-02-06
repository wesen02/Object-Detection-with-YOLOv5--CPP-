cmake_minimum_required(VERSION 3.0.2)

project(YOLOv5)

find_package(OpenCV REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(opencv src/main.cpp)
target_link_libraries(opencv ${OpenCV_LIBS})