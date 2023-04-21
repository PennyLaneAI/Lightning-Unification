cmake_minimum_required(VERSION 3.20)

project(lightning_components LANGUAGES CXX)

###############################################################################
# Include all nested sources directories
###############################################################################
set(COMPONENT_SUBDIRS      base
                           executor
                           measurements
                           simulators
                           utils
)
foreach(COMP ${COMPONENT_SUBDIRS})
    add_subdirectory(${COMP})
endforeach()

if (BUILD_TESTS)
    # Include macros supporting tests.
    include("${CMAKE_SOURCE_DIR}/cmake/support_tests.cmake")
    FetchAndIncludeCatch()

    include(CTest)
endif()