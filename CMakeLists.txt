cmake_minimum_required(VERSION 3.16)
project(nn)
 
set(CMAKE_CXX_STANDARD 17)

include_directories(src)
 
add_executable(test_armadillo tests/test_armadillo.cc)
target_link_libraries(test_armadillo armadillo)

file(GLOB_RECURSE CC_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/*.cc")

add_executable(xor_calculator tests/xor_calculator.cc
                ${CC_SOURCES})
target_link_libraries(xor_calculator armadillo)

add_executable(train_lenet tests/train_lenet.cc
                ${CC_SOURCES})
target_link_libraries(train_lenet armadillo)

