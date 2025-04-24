#pragma once
#include "DataHandling/StorageManager.hpp"
#include "QueryProcessing/QueryExecutor.hpp"
#include "InputParser.hpp"
#include "ResultDisplayer.hpp"
#include <string>
#include <memory>
#include <vector>

namespace SQLQueryProcessor {

class CommandLineInterface {
public:
    CommandLineInterface(StorageManager& storageManager);
    ~CommandLineInterface() = default;
    
    // Start the CLI
    void run();
    
    // Process a single query
    bool processQuery(const std::string& query);
    
    // Execute a query from file
    bool executeQueryFile(const std::string& filePath);
    
    // Set execution mode
    void setExecutionMode(QueryExecutor::ExecutionMode mode);
    
    // Enable/disable streaming
    void enableStreaming(bool enable);
    
private:
    StorageManager& storageManager;
    QueryExecutor queryExecutor;
    InputParser inputParser;
    ResultDisplayer resultDisplayer;
    bool exitRequested;
    std::string lastError;
    
    // Command handlers
    void handleHelpCommand();
    void handleListTablesCommand();
    void handleDescribeTableCommand(const std::string& tableName);
    void handleShowTableCommand(const std::string& tableName);
    void handleModeCommand(const std::string& mode);
    void handleStreamingCommand(const std::string& enable);
    void handleExitCommand();
    void handleSQLQuery(const std::string& query);
    
    // Display query execution statistics
    void displayStats(const QueryExecutor::ExecutionStats& stats);
};

} // namespace SQLQueryProcessor