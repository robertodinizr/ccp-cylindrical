cmake_minimum_required(VERSION 3.20)
project(ccp-cylindrical LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

set(CPM_DOWNLOAD_VERSION 0.40.2)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake...")
    file(DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()
include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage("gh:d99kris/rapidcsv@8.83")
CPMAddPackage("gh:p-ranav/argparse@3.1")

include(FetchContent)
FetchContent_Declare(
  hypre
  GIT_REPOSITORY https://github.com/hypre-space/hypre.git
  GIT_TAG v2.28.0
)
FetchContent_MakeAvailable(hypre)

add_subdirectory(spark)

target_include_directories(spark PUBLIC
    "${hypre_SOURCE_DIR}/src/HYPRE/include"
    "${hypre_SOURCE_DIR}/src/struct_ls"
    "${hypre_SOURCE_DIR}/src/utilities"
)

target_include_directories(spark PUBLIC
    "${hypre_BINARY_DIR}"
)

target_link_libraries(spark PUBLIC HYPRE)

set(SOURCES
    src/main.cpp
    src/simulation.cpp
    src/parameters.cpp
    src/reactions.cpp
    src/simulation_events.cpp
)

add_executable(ccp-cylindrical ${SOURCES})

target_link_libraries(ccp-cylindrical PUBLIC
    spark
    MPI::MPI_CXX
    rapidcsv
    argparse
)