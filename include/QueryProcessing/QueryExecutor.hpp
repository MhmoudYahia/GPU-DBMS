#pragma once
#include "DataHandling/Table.hpp"
#include "DataHandling/StorageManager.hpp"
#include "Operations/Select.hpp"
#include "Operations/Project.hpp"
#include "Operations/Join.hpp"
#include "Operations/Filter.hpp"
#include "Operations/OrderBy.hpp"
#include "Operations/Aggregator.hpp"
#include <memory>
#include <string>
#include <vector>
#include <chrono>

namespace SQLQueryProcessor {

class QueryExecutor {
public:
    enum class ExecutionMode {
        CPU,
        GPU,
        AUTO  // Automatically choose between CPU and GPU based on data size
    };
    
    struct ExecutionStats {
        double totalTimeMs;
        double parsingTimeMs;
        double executionTimeMs;
        double outputTimeMs;
        size_t resultSize;
        
        ExecutionStats()
            : totalTimeMs(0), parsingTimeMs(0), executionTimeMs(0), outputTimeMs(0), resultSize(0) {}
    };
    
    QueryExecutor(StorageManager& storageManager, ExecutionMode mode = ExecutionMode::AUTO);
    ~QueryExecutor() = default;
    
    // Execute a SQL query and return the result
    std::shared_ptr<Table> executeQuery(const std::string& query, ExecutionStats& stats);
    
    // Set execution mode
    void setExecutionMode(ExecutionMode mode);
    ExecutionMode getExecutionMode() const;
    
    // Enable/disable streaming
    void enableStreaming(bool enable);
    bool isStreamingEnabled() const;
    
private:
    StorageManager& storageManager;
    ExecutionMode executionMode;
    bool streamingEnabled;
    
    // Operation objects
    Select selectOp;
    Project projectOp;
    Join joinOp;
    Filter filterOp;
    OrderBy orderByOp;
    Aggregator aggregatorOp;
    
    // Helper methods for query execution
    std::shared_ptr<Table> executeSubquery(const std::string& subquery);
    ExecutionMode determineOptimalMode(size_t dataSize);
};

} // namespace SQLQueryProcessor