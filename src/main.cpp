#include "CLI/CommandLineInterface.hpp"
#include "DataHandling/StorageManager.hpp"
#include "Utilities/Logger.hpp"
#include "Utilities/ErrorHandling.hpp"
#include <iostream>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace SQLQueryProcessor;

// Function to check CUDA device properties
bool checkCUDACapabilities() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        Logger::warning("CUDA initialization failed: " + std::string(cudaGetErrorString(error)));
        return false;
    }
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        Logger::warning("No CUDA-capable devices found");
        return false;
    }
    
    // Get properties for the first device
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    
    Logger::info("Using CUDA Device #0: " + std::string(deviceProp.name));
    Logger::info("Compute Capability: " + std::to_string(deviceProp.major) + "." + std::to_string(deviceProp.minor));
    Logger::info("Total Global Memory: " + std::to_string(deviceProp.totalGlobalMem / (1024 * 1024)) + " MB");
    
    return true;
}

int main(int argc, char* argv[]) {
    try {
        // Initialize logger
        Logger::init("sql_processor.log");
        Logger::setLogLevel(LogLevel::INFO);
        Logger::enableConsoleOutput(false);
        Logger::info("SQLQueryProcessor starting up...");
        
        // Check CUDA capabilities
        bool cudaAvailable = checkCUDACapabilities();
        if (!cudaAvailable) {
            Logger::warning("Running without CUDA acceleration");
        }
        
        // Parse command line arguments
        InputParser inputParser;
        inputParser.parseArgs(argc, argv);
        
        // Initialize storage manager with data directory
        std::string dataDir = inputParser.getOption("--data-dir");
        if (dataDir.empty()) {
            dataDir = "data/input_csvs/";
        }
        
        StorageManager storageManager(dataDir);
        Logger::info("Loading tables from " + dataDir);
        storageManager.loadTables();
        
        // Initialize CLI
        CommandLineInterface cli(storageManager);
        
        // Set output mode if specified
        std::string outputMode = inputParser.getOption("--output-mode");
        if (outputMode == "console") {
            cli.setOutputMode(CommandLineInterface::OutputMode::CONSOLE_ONLY);
        } else if (outputMode == "file") {
            cli.setOutputMode(CommandLineInterface::OutputMode::FILE_ONLY);
        } else if (outputMode == "both") {
            cli.setOutputMode(CommandLineInterface::OutputMode::BOTH);
        }
        
        // Check if a query file was provided
        if (inputParser.hasFlag("--query-file")) {
            std::string queryFile = inputParser.getOption("--query-file");
            if (!queryFile.empty()) {
                Logger::info("Executing queries from file: " + queryFile);
                cli.processQuery(".execute " + queryFile, QueryExecutor::ExecutionStats());
                return 0;
            }
        }
        
        // Check if a direct query was provided
        if (inputParser.hasFlag("--query")) {
            std::string query = inputParser.getOption("--query");
            if (!query.empty()) {
                Logger::info("Executing query: " + query);
                cli.processQuery(query, QueryExecutor::ExecutionStats());
                return 0;
            }
        }
        
        // Start interactive mode
        Logger::info("Starting interactive mode");
        cli.run();
        
        Logger::info("SQLQueryProcessor shutting down normally");
        return 0;
    } catch (const SQLQueryProcessorException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        Logger::critical("Fatal error: " + std::string(e.what()));
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        Logger::critical("Unexpected error: " + std::string(e.what()));
        return 1;
    }
}