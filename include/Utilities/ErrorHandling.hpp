#pragma once
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>

namespace SQLQueryProcessor
{

    // Custom exception classes
    class SQLQueryProcessorException : public std::runtime_error
    {
    public:
        explicit SQLQueryProcessorException(const std::string &message);
    };

    class ParsingException : public SQLQueryProcessorException
    {
    public:
        explicit ParsingException(const std::string &message);
    };

    class ExecutionException : public SQLQueryProcessorException
    {
    public:
        explicit ExecutionException(const std::string &message);
    };

    class DataException : public SQLQueryProcessorException
    {
    public:
        explicit DataException(const std::string &message);
    };

    class CUDAException : public SQLQueryProcessorException
    {
    public:
        explicit CUDAException(const std::string &message, cudaError_t error);
    };

    // Error handling utilities
    namespace ErrorHandling
    {

        // Check CUDA errors and throw exception if found
        void checkCudaError(cudaError_t error, const std::string &prefix = "CUDA Error");

        // Handle errors in a uniform way
        void logError(const std::string &message);

        // Format error messages with file and line information
        std::string formatErrorMessage(const std::string &message, const char *file, int line);

    } // namespace ErrorHandling

// Convenience macro for error formatting
#define FORMAT_ERROR(message) \
    ErrorHandling::formatErrorMessage(message, __FILE__, __LINE__)

} // namespace SQLQueryProcessor