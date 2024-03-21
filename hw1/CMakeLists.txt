cmake_minimum_required(VERSION 3.23)
project(hw1)

set(CMAKE_CXX_STANDARD 14)

add_executable(${PROJECT_NAME} algo.cpp)

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()