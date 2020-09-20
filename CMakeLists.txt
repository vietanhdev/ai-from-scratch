cmake_minimum_required(VERSION 3.10)
project(nn)
 
set(CMAKE_CXX_STANDARD 17)

# Find libraries
find_package(OpenCV REQUIRED)

include_directories(
    src
    ${OpenCV_INCLUDE_DIRS}
)
 
add_executable(test_armadillo tests/test_armadillo.cc)
target_link_libraries(test_armadillo armadillo)

file(GLOB_RECURSE CC_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/*.cc")

add_library(afs ${CC_SOURCES})
target_link_libraries(afs armadillo ${OpenCV_LIBS})

add_executable(xor_calculator tests/xor_calculator.cc
                ${CC_SOURCES})
target_link_libraries(xor_calculator afs)

add_executable(wine_quality_estimator tests/wine_quality_estimator.cc
                ${CC_SOURCES})
target_link_libraries(wine_quality_estimator afs)

add_executable(wine_quality_estimator_with_dropout tests/wine_quality_estimator_with_dropout.cc
                ${CC_SOURCES})
target_link_libraries(wine_quality_estimator_with_dropout afs)

add_executable(train_lenet tests/train_lenet.cc
                ${CC_SOURCES})
target_link_libraries(train_lenet afs)

