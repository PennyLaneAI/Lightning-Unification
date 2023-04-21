cmake_minimum_required(VERSION 3.20)

project(lightning_utils LANGUAGES CXX)

add_library(lightning_utils INTERFACE)

target_include_directories(lightning_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_utils INTERFACE lightning_compile_options
                                                lightning_external_libs
                                                )

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()