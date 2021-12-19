find_package(OpenCV REQUIRED CONFIG)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(DIPCorr2D DIPCorr2D.cpp)
target_link_libraries(DIPCorr2D ${OpenCV_LIBS} Corr2D Container PNGImage)
