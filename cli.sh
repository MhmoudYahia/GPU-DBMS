#!/bin/bash
# filepath: /mnt/g/MyRepos/SQLQueryProcessor/run.sh

# Default settings
DATA_DIR="/mnt/g/MyRepos/SQLQueryProcessor/data"
MODE="cli"
TEST_NAME=""
GPU_MODE="on"

# Function to show help
show_help() {
    echo "SQL Query Processor"
    echo "Usage: ./run.sh [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -d, --data-dir <dir>      Specify data directory (default: ./data)"
    echo "  -t, --test <test_name>    Run specific test"
    echo "  --test-all                Run all tests"
    echo "  --gpu <on|off>            Enable/disable GPU execution"
    echo
    echo "Available tests: select, project, condition, orderby, aggregate,"
    echo "                join, sql, csv, datetime, boolean"
    echo
    echo "Examples:"
    echo "  ./run.sh                  # Run CLI with default data directory"
    echo "  ./run.sh -d /path/to/data # Run CLI with custom data directory"
    echo "  ./run.sh -t select        # Run select test"
    echo "  ./run.sh --test-all       # Run all tests"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--data-dir)
            if [[ $# -gt 1 ]]; then
                DATA_DIR="$2"
                shift
            else
                echo "Error: --data-dir requires a directory path"
                exit 1
            fi
            ;;
        -t|--test)
            MODE="test"
            if [[ $# -gt 1 && ! $2 =~ ^- ]]; then
                TEST_NAME="$2"
                shift
            else
                echo "Error: --test requires a test name"
                exit 1
            fi
            ;;
        --test-all)
            MODE="test"
            TEST_NAME="all"
            ;;
        --gpu)
            if [[ $# -gt 1 && ($2 == "on" || $2 == "off") ]]; then
                GPU_MODE="$2"
                shift
            else
                echo "Error: --gpu requires 'on' or 'off'"
                exit 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
    shift
done

# Create build directory if it doesn't exist
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

# Navigate to build directory
cd "$BUILD_DIR"

# Generate build files
cmake ..

# Build the project
echo "Building SQLQueryProcessor..."
cmake --build . -- -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
    
    # Run the executable based on mode
    if [ "$MODE" == "test" ]; then
        echo "Running test: $TEST_NAME"
        ./bin/GPUDBMS --test "$TEST_NAME" --gpu "$GPU_MODE"
    else
        echo "Starting CLI with data directory: $DATA_DIR"
        ./bin/GPUDBMS --data-dir "$DATA_DIR" --gpu "$GPU_MODE"
    fi
else
    echo "Build failed. Please check the errors above."
    exit 1
fi