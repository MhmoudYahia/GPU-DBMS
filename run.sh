#!/bin/bash

# Build directory
BUILD_DIR="build"

# Create build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

# Navigate to build directory
cd "$BUILD_DIR"

# Generate build files with ASAN enabled
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address -g -O1" -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" ..

# Build the project
cmake --build . -- -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful! Running GPUDBMS with AddressSanitizer..."

    # Run the executable with any passed arguments
    ASAN_OPTIONS=detect_leaks=1 ./bin/GPUDBMS "$@"
else
    echo "Build failed. Please check the errors above."
    exit 1
fi
