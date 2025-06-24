cmake_minimum_required(VERSION 3.20)
project(ccp-cylindrical LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_path(HYPRE_INCLUDE_DIR NAMES HYPRE.h PATH_SUFFIXES hypre)
find_library(HYPRE_LIBRARY NAMES HYPRE)
if(HYPRE_INCLUDE_DIR AND HYPRE_LIBRARY)
    add_library(HYPRE::HYPRE INTERFACE IMPORTED)
    set_target_properties(HYPRE::HYPRE PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${HYPRE_LIBRARY}"
    )
    message(STATUS "HYPRE encontrado manualmente em ${HYPRE_LIBRARY}")
else()
    message(FATAL_ERROR "Não foi possível encontrar a biblioteca HYPRE manualmente.")
endif()

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

add_subdirectory(spark)

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
    HYPRE::HYPRE
    MPI::MPI_CXX
    rapidcsv
    argparse
)
