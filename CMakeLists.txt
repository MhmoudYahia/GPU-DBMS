cmake_minimum_required(VERSION 3.10)

# Force compiler selection before project() declaration
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

# Set the project and enable CUDA
project(GPUDBMS LANGUAGES CXX CUDA)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check if CUDA is available
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_AVAILABLE TRUE)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61")
    add_compile_definitions(GPU_AVAILABLE)

    # Explicitly add CUDA include path
    include_directories(/usr/local/cuda-12.8/targets/x86_64-linux/include)
else()
    set(CUDA_AVAILABLE FALSE)
    message(WARNING "CUDA not found, GPU functionality will be disabled.")
endif()

# Include project directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

if(CUDA_AVAILABLE)
    file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
    list(APPEND SOURCES ${CUDA_SOURCES})
endif()

# Add SQLProcessing GPU source only if SQL parser is found (later)
set(SQL_PROCESSING_SRC src/SQLProcessing/GPUDBMS.cpp)

# Create the executable
add_executable(GPUDBMS ${SOURCES})

# CUDA settings
if(CUDA_AVAILABLE)
    target_link_libraries(GPUDBMS PRIVATE cudart)
    target_compile_definitions(GPUDBMS PRIVATE GPU_AVAILABLE)
    set_target_properties(GPUDBMS PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Set output directory
set_target_properties(GPUDBMS 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Installation
install(TARGETS GPUDBMS DESTINATION bin)

# Enable testing
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    enable_testing()
    add_subdirectory(tests)
endif()

# SQL Parser Detection
find_path(SQLPARSER_INCLUDE_DIR hsql/SQLParser.h
  HINTS /usr/local/include)
find_library(SQLPARSER_LIBRARY sqlparser
  HINTS /usr/local/lib)

if(SQLPARSER_INCLUDE_DIR AND SQLPARSER_LIBRARY)
    message(STATUS "SQL Parser found: ${SQLPARSER_LIBRARY}")
    include_directories(${SQLPARSER_INCLUDE_DIR})
    target_sources(GPUDBMS PRIVATE ${SQL_PROCESSING_SRC})
    target_link_libraries(GPUDBMS PRIVATE ${SQLPARSER_LIBRARY})
else()
    message(FATAL_ERROR "SQL Parser not found. Please install it or specify its location.")
endif()
