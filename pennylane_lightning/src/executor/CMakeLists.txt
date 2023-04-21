cmake_minimum_required(VERSION 3.20)

project(lightning_executor LANGUAGES CXX)

add_library(lightning_executor INTERFACE)

target_include_directories(lightning_executor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lightning_executor INTERFACE  lightning_compile_options
                                                    lightning_external_libs
                                                    )

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()