cmake_minimum_required(VERSION 3.20)

project(pl_qubit_utils LANGUAGES CXX)

add_library(pl_qubit_utils INTERFACE)

target_include_directories(pl_qubit_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pl_qubit_utils INTERFACE  lightning_utils)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()