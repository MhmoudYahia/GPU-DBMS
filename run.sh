#!/bin/bash

set -e  # Exit immediately on error

# Build directory
BUILD_DIR="build"

# Create build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

# Navigate to build directory
cd "$BUILD_DIR"

echo "Running CMake..."
cmake .. || { echo "CMake configuration failed."; exit 1; }

echo "Building project..."
cmake --build . -- -j$(nproc) || { echo "Build failed."; exit 1; }

echo "Build successful! Running GPUDBMS..."
./bin/GPUDBMS "$@"
