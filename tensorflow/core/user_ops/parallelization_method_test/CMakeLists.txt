cmake_minimum_required(VERSION 2.8)

add_definitions(-std=gnu++0x)
add_definitions(-DBUILD_WITHOUT_LEICA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-long-long -fopenmp")

project(performance_evaluation_methods)

set( PROJECT_LINK_LIBS libcityhash.so )
find_package(Boost COMPONENTS system thread REQUIRED)
include_directories( ${Boost_INCLUDE_DIR} )

include_directories(time)
add_library(time time/time.cpp)

add_executable(merge_sort merge_sort.cpp)

add_executable(kd_map kd_map.cpp)
target_link_libraries(kd_map ${Boost_LIBRARIES} time ${PROJECT_LINK_LIBS})
