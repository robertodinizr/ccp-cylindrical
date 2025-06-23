cmake_minimum_required(VERSION 3.20)
project(ccp-cylindrical)

set(CMAKE_CXX_STANDARD 20)
set(FETCHCONTENT_QUIET OFF CACHE BOOL "" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Add CPM
set(CPM_DOWNLOAD_VERSION 0.40.2)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake...")
    file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()
include(${CPM_DOWNLOAD_LOCATION})

# Add external libs
CPMAddPackage("gh:d99kris/rapidcsv@8.83")
CPMAddPackage("gh:p-ranav/argparse@3.1")
find_package(hypre CONFIG REQUIRED)
find_package(MPI REQUIRED)
find_package(rapidcsv CONFIG REQUIRED)
find_package(argparse CONFIG REQUIRED)
add_subdirectory(spark)
include(FetchContent)

FetchContent_Declare(
  hypre
  GIT_REPOSITORY https://github.com/hypre-space/hypre.git
  GIT_TAG v2.28.0  
  CMAKE_ARGS "-DBUILD_SHARED_LIBS=OFF"
)
FetchContent_MakeAvailable(hypre)

set(SOURCES
    src/main.cpp
    src/simulation.cpp
    src/parameters.cpp
    src/reactions.cpp
    src/simulation_events.cpp
)
find_package(MPI REQUIRED)
add_executable(ccp-cylindrical ${SOURCES})
set_property(TARGET ccp-cylindrical PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

target_include_directories(ccp-cylindrical PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/spark/include
    ${argparse_SOURCE_DIR}/include
    ${rapidcsv_SOURCE_DIR}/src
    ${MPI_CXX_INCLUDE_PATH}
)

target_link_libraries(ccp-cylindrical PUBLIC
    spark
    HYPRE
    MPI::MPI_CXX
    rapidcsv
    argparse
)
