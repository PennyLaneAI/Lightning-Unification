cmake_minimum_required(VERSION 3.20)

set(LOGO [=[
░█░░░▀█▀░█▀▀░█░█░▀█▀░█▀█░▀█▀░█▀█░█▀▀░░░░▄▀▄░█░█░█▀▄░▀█▀░▀█▀
░█░░░░█░░█░█░█▀█░░█░░█░█░░█░░█░█░█░█░░░░█\█░█░█░█▀▄░░█░░░█░
░▀▀▀░▀▀▀░▀▀▀░▀░▀░░▀░░▀░▀░▀▀▀░▀░▀░▀▀▀░▀░░░▀\░▀▀▀░▀▀░░▀▀▀░░▀░
]=])
message(${LOGO})

project(lightning_qubit
        DESCRIPTION "PennyLane Lightning C++ Backend."
        LANGUAGES CXX C
)

set(LQUBIT_FILES    StateVectorLQubitManaged.cpp
                    StateVectorLQubitRaw.cpp
                    CACHE INTERNAL "" FORCE)

add_library(lightning_qubit STATIC ${LQUBIT_FILES})

option(ENABLE_KOKKOS "Enable Kokkos" ON)
option(ENABLE_BLAS "Enable BLAS" OFF)
option(ENABLE_GATE_DISPATCHER "Enable gate kernel dispatching on AVX/AVX2/AVX512" ON)

# Inform the compiler that this device is enabled.
target_compile_options(lightning_compile_options INTERFACE "-D_ENABLE_PLQUBIT=1")

if(ENABLE_KOKKOS)
    # Include macro and functions supporting Kokkos libraries.
    include("${CMAKE_SOURCE_DIR}/cmake/support_kokkos.cmake")

    message(STATUS "ENABLE_KOKKOS is ON.")
    target_compile_options(lightning_compile_options INTERFACE "-D_ENABLE_KOKKOS=1")
    FindKokkosAndKokkosKernels(lightning_external_libs)
else()
    message(STATUS "ENABLE_KOKKOS is OFF.")
endif()

if(ENABLE_BLAS)
    message(STATUS "ENABLE_BLAS is ON.")
    # find_package(BLAS)

    # if(NOT BLAS_FOUND)
    #     message(FATAL_ERROR "BLAS is enabled but not found.")
    # endif()
    find_package(MKL QUIET)

    if(MKL_FOUND)
        add_definitions("-DENABLE_MKL")
        set(BLAS_INCLUDE_DIRS "${MKL_INCLUDE_DIR}")
        set(BLAS_LIBRARIES ${MKL_LIBRARY})
    else()
        find_package(CBLAS REQUIRED)
        set(BLAS_INCLUDE_DIRS ${CBLAS_INCLUDE_DIRS})
        set(BLAS_LIBRARIES ${CBLAS_LIBRARIES})
    endif()

    # get_blas_impl()
    # message(STATUS "Use ${BLAS_IMPL} for BLAS implementation. Set BLA_VENDOR variable "
    #                "if you want to use a different BLAS implementation. "
    #                "See https://cmake.org/cmake/help/latest/module/FindBLAS.html"
    #                "#blas-lapack-vendors for available options.")

    target_link_libraries(lightning_external_libs INTERFACE "${BLAS_LIBRARIES}")
    # target_link_options(lightning_external_libs INTERFACE "${BLAS_LINKER_FLAGS}")
    target_include_directories(lightning_external_libs INTERFACE "${BLAS_INCLUDE_DIRS}")
    target_compile_options(lightning_compile_options INTERFACE "-D_ENABLE_BLAS=1")
else()
    message(STATUS "ENABLE_BLAS is OFF.")
endif()

target_link_libraries(lightning_qubit PUBLIC    lightning_compile_options
                                                lightning_external_libs
                                                lightning_base
                                                lightning_utils
                                                pl_qubit_utils
                                                pl_qubit_gates
                                                )

target_include_directories(lightning_qubit PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET lightning_qubit PROPERTY POSITION_INDEPENDENT_CODE ON)

###############################################################################
# Include subdirectories
###############################################################################
set(COMPONENT_SUBDIRS   gates;
                        utils;
                        )

foreach(COMP ${COMPONENT_SUBDIRS})
    add_subdirectory(${COMP})
endforeach()

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()
