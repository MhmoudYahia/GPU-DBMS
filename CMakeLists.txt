cmake_minimum_required(VERSION 3.10)

# Force compiler selection before project() declaration
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

# Check if CUDA is available
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    project(SQLQueryProcessor LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61")
    set(CUDA_AVAILABLE TRUE)
else()
    project(SQLQueryProcessor LANGUAGES CXX)
    set(CUDA_AVAILABLE FALSE)
    message(WARNING "CUDA not found, GPU functionality will be disabled.")
endif()

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs)

# Add source files
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
)

if(CUDA_AVAILABLE)
    file(GLOB_RECURSE CUDA_SOURCES 
        "src/*.cu"
    )
    list(APPEND SOURCES ${CUDA_SOURCES})
endif()

# Create executable
add_executable(sqlqueryprocessor ${SOURCES})

# Link libraries
if(CUDA_AVAILABLE)
    target_link_libraries(sqlqueryprocessor PRIVATE cudart)
    target_compile_definitions(sqlqueryprocessor PRIVATE GPU_AVAILABLE)
endif()

# Set output directory
set_target_properties(sqlqueryprocessor 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

if(CUDA_AVAILABLE)
    set_target_properties(sqlqueryprocessor PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# Installation
install(TARGETS sqlqueryprocessor DESTINATION bin)

# Enable testing if tests directory exists
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    enable_testing()
    add_subdirectory(tests)
endif()


# Find SQL Parser package
find_path(SQLPARSER_INCLUDE_DIR hsql/SQLParser.h
  HINTS /usr/local/include)
find_library(SQLPARSER_LIBRARY sqlparser
  HINTS /usr/local/lib)

include_directories(${SQLPARSER_INCLUDE_DIR})

# Add the new SQLProcessing sources
set(SOURCES 
    ${SOURCES}
    src/SQLProcessing/SQLQueryProcessor.cpp)

# Link against the SQL parser library
target_link_libraries(sqlqueryprocessor ${SQLPARSER_LIBRARY})