#!/bin/bash

# Build directory
BUILD_DIR="build"

# Create build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

# Navigate to build directory
cd "$BUILD_DIR"

# Generate build files
cmake ..

# Build the project
cmake --build . -- -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful! Running SQLQueryProcessor..."
    
    # Run the executable
    ./bin/sqlqueryprocessor "$@"
else
    echo "Build failed. Please check the errors above."
    exit 1
fi