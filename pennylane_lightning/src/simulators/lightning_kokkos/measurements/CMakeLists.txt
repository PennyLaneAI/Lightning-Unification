cmake_minimum_required(VERSION 3.20)

project(${PL_BACKEND}_measurements LANGUAGES CXX)

set(OBSERVABLES_FILES MeasuresKokkos.cpp CACHE INTERNAL "" FORCE)
add_library(${PL_BACKEND}_measurements STATIC ${OBSERVABLES_FILES})

target_include_directories(${PL_BACKEND}_measurements INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${PL_BACKEND}_measurements INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PL_BACKEND}_measurements PRIVATE    lightning_compile_options
                                                            lightning_external_libs
                                                            )

target_link_libraries(${PL_BACKEND}_measurements PUBLIC     
                                                                lightning_measurements
                                                                lightning_observables
                                                                lightning_utils
                                                                ${PL_BACKEND}
                                                                ${PL_BACKEND}_utils
                                                            )

set_property(TARGET ${PL_BACKEND}_measurements PROPERTY POSITION_INDEPENDENT_CODE ON)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()