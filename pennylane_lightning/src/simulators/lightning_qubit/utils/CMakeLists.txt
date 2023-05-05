cmake_minimum_required(VERSION 3.20)

project(pl_qubit_utils LANGUAGES CXX)

set(LQUBIT_UTILS_FILES RuntimeInfo.cpp CACHE INTERNAL "" FORCE)
add_library(pl_qubit_utils STATIC ${LQUBIT_UTILS_FILES})

target_include_directories(pl_qubit_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pl_qubit_utils INTERFACE  lightning_compile_options
                                                lightning_external_libs
                                                lightning_utils
                                                )

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()