#include "QueryProcessing/QueryExecutor.hpp"
#include "QueryProcessing/ASTProcessor.hpp"
#include "QueryProcessing/PlanBuilder.hpp"
#include "Utilities/Logger.hpp"
#include "Utilities/ErrorHandling.hpp"
#include <chrono>

namespace SQLQueryProcessor {

QueryExecutor::QueryExecutor(StorageManager& storageManager, ExecutionMode mode)
    : storageManager(storageManager), 
      executionMode(mode),
      streamingEnabled(false) {
}

std::shared_ptr<Table> QueryExecutor::executeQuery(const std::string& query, ExecutionStats& stats) {
    Logger::info("Executing query: " + query);
    
    // Reset statistics
    stats = ExecutionStats();
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        // Parse the query and build execution plan
        auto parsingStart = std::chrono::high_resolution_clock::now();
        
        ASTProcessor astProcessor(storageManager);
        ASTProcessor::QueryInfo queryInfo = astProcessor.processQuery(query);
        
        auto parsingEnd = std::chrono::high_resolution_clock::now();
        stats.parsingTimeMs = std::chrono::duration<double, std::milli>(parsingEnd - parsingStart).count();
        
        // Execute the query
        auto executionStart = std::chrono::high_resolution_clock::now();
        
        // Determine whether to use GPU
        bool useGPU = (executionMode == ExecutionMode::GPU) ||
                     (executionMode == ExecutionMode::AUTO && 
                      determineOptimalMode(0) == ExecutionMode::GPU);  // Size determination would happen in the real implementation
        
        PlanBuilder planBuilder(storageManager);
        std::shared_ptr<Table> resultTable = planBuilder.buildAndExecutePlan(
            queryInfo, useGPU, streamingEnabled);
        
        auto executionEnd = std::chrono::high_resolution_clock::now();
        stats.executionTimeMs = std::chrono::duration<double, std::milli>(executionEnd - executionStart).count();
        
        // Output processing
        auto outputStart = std::chrono::high_resolution_clock::now();
        
        // Here we would normally format and return results
        // For now, just record the result size
        if (resultTable) {
            stats.resultSize = resultTable->getRowCount();
        }
        
        auto outputEnd = std::chrono::high_resolution_clock::now();
        stats.outputTimeMs = std::chrono::duration<double, std::milli>(outputEnd - outputStart).count();
        
        // Calculate total time
        auto endTime = std::chrono::high_resolution_clock::now();
        stats.totalTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        
        Logger::info("Query executed successfully in " + std::to_string(stats.totalTimeMs) + " ms");
        return resultTable;
        
    } catch (const std::exception& e) {
        auto endTime = std::chrono::high_resolution_clock::now();
        stats.totalTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        
        Logger::error("Query execution failed: " + std::string(e.what()));
        throw;
    }
}

void QueryExecutor::setExecutionMode(ExecutionMode mode) {
    executionMode = mode;
    Logger::info("Execution mode set to " + 
                (mode == ExecutionMode::CPU ? "CPU" : 
                 (mode == ExecutionMode::GPU ? "GPU" : "AUTO")));
}

QueryExecutor::ExecutionMode QueryExecutor::getExecutionMode() const {
    return executionMode;
}

void QueryExecutor::enableStreaming(bool enable) {
    streamingEnabled = enable;
    Logger::info(std::string("CUDA streaming ") + (enable ? "enabled" : "disabled"));
}

bool QueryExecutor::isStreamingEnabled() const {
    return streamingEnabled;
}

std::shared_ptr<Table> QueryExecutor::executeSubquery(const std::string& subquery) {
    ExecutionStats stats;
    return executeQuery(subquery, stats);
}

QueryExecutor::ExecutionMode QueryExecutor::determineOptimalMode(size_t dataSize) {
    // This would use heuristics based on data size, operation complexity, etc.
    // For now, use a simple size-based heuristic
    if (dataSize > 1000000) {  // 1M rows
        return ExecutionMode::GPU;
    } else {
        return ExecutionMode::CPU;
    }
}

} // namespace SQLQueryProcessor