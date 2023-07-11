project(${PL_BACKEND}_utils LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_library(${PL_BACKEND}_utils INTERFACE)
# target_include_directories(${PL_BACKEND}_utils INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
#                                                      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/DS>
# )
target_include_directories(${PL_BACKEND}_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PL_BACKEND}_utils INTERFACE  lightning_utils)
