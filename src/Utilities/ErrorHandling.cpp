#include "Utilities/ErrorHandling.hpp"
#include "Utilities/Logger.hpp"
#include <sstream>

namespace SQLQueryProcessor {

// Exception implementations
SQLQueryProcessorException::SQLQueryProcessorException(const std::string& message)
    : std::runtime_error(message) {
    // Log the exception
    Logger::error("Exception: " + message);
}

ParsingException::ParsingException(const std::string& message)
    : SQLQueryProcessorException("Parsing error: " + message) {
}

ExecutionException::ExecutionException(const std::string& message)
    : SQLQueryProcessorException("Execution error: " + message) {
}

DataException::DataException(const std::string& message)
    : SQLQueryProcessorException("Data error: " + message) {
}

CUDAException::CUDAException(const std::string& message, cudaError_t error)
    : SQLQueryProcessorException(message + ": " + cudaGetErrorString(error)) {
}

namespace ErrorHandling {

void checkCudaError(cudaError_t error, const std::string& prefix) {
    if (error != cudaSuccess) {
        std::string errorMessage = prefix + ": " + cudaGetErrorString(error);
        Logger::error(errorMessage);
        throw CUDAException(prefix, error);
    }
}

void logError(const std::string& message) {
    Logger::error(message);
}

std::string formatErrorMessage(const std::string& message, const char* file, int line) {
    std::ostringstream oss;
    oss << message << " [" << file << ":" << line << "]";
    return oss.str();
}

} // namespace ErrorHandling

} // namespace SQLQueryProcessor