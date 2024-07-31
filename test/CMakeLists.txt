cmake_minimum_required(VERSION 3.18)
project(UnitTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)
find_package(CUDA REQUIRED)

set(link_ext_lib GTest::gtest)

aux_source_directory(../test DIR_TEST)
aux_source_directory(../src DIR_SRC)

include_directories(../include)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(unit_test ${DIR_TEST} ${DIR_SRC})
target_link_libraries(unit_test ${link_ext_lib} ${CUDA_LIBRARIES})
