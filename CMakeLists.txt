cmake_minimum_required(VERSION 3.10)

# Optional: Force specific compilers
# For MSVC (Visual Studio) or MinGW, no need to specify compilers unless using a specific one.
# For MinGW, you can specify paths to gcc and g++ if needed.

# Set compilers for MinGW or MSVC (modify based on your setup)
# For MSVC, use the default compiler; for MinGW, specify paths as necessary.
# set(CMAKE_C_COMPILER "C:/MinGW/bin/gcc.exe")
# set(CMAKE_CXX_COMPILER "C:/MinGW/bin/g++.exe")

# Check for CUDA support
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_AVAILABLE TRUE)
else()
    set(CUDA_AVAILABLE FALSE)
endif()

project(GPUDBMS LANGUAGES CXX CUDA)

# Set CMake flags for CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --ptxas-options=-v")

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CUDA_AVAILABLE)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Target RTX 3060 (Ampere)
    set(CMAKE_CUDA_ARCHITECTURES 86)

    # Optional: use CUDA Toolkit (adjust path for Windows)
    find_package(CUDAToolkit REQUIRED)
endif()

# Include directories (adjust CUDA path for Windows)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/libs
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/include"  # Adjust based on CUDA version installed
)

# Gather sources
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")

# Create executable with explicit CUDA sources
add_executable(GPUDBMS ${CPP_SOURCES} ${CUDA_SOURCES})

# Link CUDA runtime
if(CUDA_AVAILABLE)
    target_link_libraries(GPUDBMS PRIVATE CUDA::cudart)
    target_compile_definitions(GPUDBMS PRIVATE GPU_AVAILABLE)
    set_target_properties(GPUDBMS PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# SQL Parser integration (adjust paths for Windows)
find_path(SQLPARSER_INCLUDE_DIR "hsql/SQLParser.h" HINTS "C:/path/to/sqlparser/include")
find_library(SQLPARSER_LIBRARY sqlparser HINTS "C:/path/to/sqlparser/lib")

if(SQLPARSER_INCLUDE_DIR AND SQLPARSER_LIBRARY)
    message(STATUS "SQL Parser found: ${SQLPARSER_LIBRARY}")
    include_directories(${SQLPARSER_INCLUDE_DIR})
    target_link_libraries(GPUDBMS PRIVATE ${SQLPARSER_LIBRARY})
else()
    message(FATAL_ERROR "SQL Parser not found.")
endif()

# Output path
set_target_properties(GPUDBMS PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install target (optional on Windows)
# You may skip this on Windows or use a different installation method
# install(TARGETS GPUDBMS DESTINATION bin)

# Enable testing if tests exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    enable_testing()
    add_subdirectory(tests)
endif()
