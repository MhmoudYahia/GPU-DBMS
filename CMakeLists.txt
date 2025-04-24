cmake_minimum_required(VERSION 3.10)
project(SQLQueryProcessor LANGUAGES CXX CUDA)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Settings
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs)

# Add source files
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "src/*.cu"
)

# Create executable
add_executable(sqlqueryprocessor ${SOURCES})

# Link libraries
target_link_libraries(sqlqueryprocessor PRIVATE cudart)

# Set output directory
set_target_properties(sqlqueryprocessor 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    CUDA_SEPARABLE_COMPILATION ON
)

# Installation
install(TARGETS sqlqueryprocessor DESTINATION bin)

# Enable testing if tests directory exists
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    enable_testing()
    add_subdirectory(tests)
endif()