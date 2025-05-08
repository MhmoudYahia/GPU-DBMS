#pragma once

#include <string>
#include <memory>
#include <vector>
#include <chrono>
#include "../DataHandling/Table.hpp"
#include "ResultDisplayer.hpp"

namespace GPUDBMS {

// Forward declarations
class SQLQueryProcessor;

/**
 * @class CommandLineInterface
 * @brief Provides a command-line interface for interacting with the SQL Query Processor
 */
class CommandLineInterface {
public:
    /**
     * @brief Constructor
     * @param dataDirectory Directory where database files are stored
     */
    CommandLineInterface(const std::string& dataDirectory);
    
    /**
     * @brief Destructor
     */
    ~CommandLineInterface();
    
    /**
     * @brief Start the command-line interface
     */
    void run();

private:
    // Processor and state
    std::unique_ptr<SQLQueryProcessor> m_processor;
    bool m_quit;
    bool m_useGPU = true;
    bool m_showTiming = true;
    Table m_lastResult;
    std::vector<std::string> m_availableTables;
    
    // UI methods
    void printWelcomeMessage();
    void processCommand(const std::string& command);
    void processSpecialCommand(const std::string& command);
    void executeQuery(const std::string& query);
    void displayResultTable(const Table& table);
    
    // Command implementations
    void showTables();
    void describeTable(const std::string& tableName);
    void saveResultToCSV(const std::string& filename);
    void loadAndExecuteSQL(const std::string& filename);
    
    // Helper methods
    std::string trimString(const std::string& str);
};

} // namespace GPUDBMS