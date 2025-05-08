cmake_minimum_required(VERSION 3.10)

# # Force compiler selection before project() declaration
# set(CMAKE_C_COMPILER "/usr/bin/gcc")
# set(CMAKE_CXX_COMPILER "/usr/bin/g++")

# Set project first to avoid re-evaluation issues
project(GPUDBMS LANGUAGES CXX CUDA)

# Check if CUDA is available
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61 -Wno-deprecated-gpu-targets")
    set(CUDA_AVAILABLE TRUE)
    add_compile_definitions(GPU_AVAILABLE)
    message(STATUS "CUDA found, enabling GPU functionality.")
else()
    set(CUDA_AVAILABLE FALSE)
    message(STATUS "CUDA not found, GPU functionality will be disabled.")
endif()

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/libs
    /usr/local/cuda-12.8/targets/x86_64-linux/include
)

# Find readline package
find_path(READLINE_INCLUDE_DIR readline/readline.h)
find_library(READLINE_LIBRARY readline)

if(READLINE_INCLUDE_DIR AND READLINE_LIBRARY)
    include_directories(${READLINE_INCLUDE_DIR})
    message(STATUS "Readline found: ${READLINE_LIBRARY}")
else()
    message(FATAL_ERROR "Readline library not found. Please install it (e.g., 'sudo apt install libreadline-dev')")
endif()

# Find SQL Parser package
find_path(SQLPARSER_INCLUDE_DIR hsql/SQLParser.h
    HINTS /usr/local/include)
find_library(SQLPARSER_LIBRARY sqlparser
    HINTS /usr/local/lib)

if(SQLPARSER_INCLUDE_DIR AND SQLPARSER_LIBRARY)
    message(STATUS "SQL Parser found: ${SQLPARSER_LIBRARY}")
    include_directories(${SQLPARSER_INCLUDE_DIR})
else()
    message(FATAL_ERROR "SQL Parser not found. Please install the SQL Parser library or specify its location.")
endif()

# Add source files
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "src/SQLProcessing/GPUDBMS.cpp"
)

if(CUDA_AVAILABLE)
    file(GLOB_RECURSE CUDA_SOURCES 
        "src/*.cu"
    )
    list(APPEND SOURCES ${CUDA_SOURCES})
endif()

# Create executable
add_executable(GPUDBMS ${SOURCES})

# Link libraries
if(CUDA_AVAILABLE)
    target_link_libraries(GPUDBMS PRIVATE cudart)
    target_compile_definitions(GPUDBMS PRIVATE GPU_AVAILABLE)
    set_target_properties(GPUDBMS PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

target_link_libraries(GPUDBMS PRIVATE 
    ${READLINE_LIBRARY}
    ${SQLPARSER_LIBRARY}
)

# Set output directory
set_target_properties(GPUDBMS 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Installation
install(TARGETS GPUDBMS DESTINATION bin)

# Enable testing if tests directory exists
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    enable_testing()
    add_subdirectory(tests)
endif()