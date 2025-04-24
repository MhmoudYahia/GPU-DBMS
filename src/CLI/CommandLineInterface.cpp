#include "CLI/CommandLineInterface.hpp"
#include "Utilities/Logger.hpp"
#include "Utilities/StringUtils.hpp"
#include <iostream>
#include <fstream>

namespace SQLQueryProcessor {

CommandLineInterface::CommandLineInterface(StorageManager& storageManager)
    : storageManager(storageManager),
      queryExecutor(storageManager, QueryExecutor::ExecutionMode::AUTO),
      resultDisplayer(storageManager),
      outputMode(OutputMode::BOTH),
      running(false) {
    
    // Initialize command handlers
    initializeCommandHandlers();
    
    Logger::debug("CommandLineInterface initialized");
}

void CommandLineInterface::run() {
    running = true;
    
    // Print welcome message
    std::cout << "SQLQueryProcessor Command Line Interface" << std::endl;
    std::cout << "Type '.help' for a list of commands or enter SQL queries" << std::endl;
    std::cout << "Type '.exit' to quit" << std::endl << std::endl;
    
    while (running) {
        std::string input = getInput();
        
        if (input.empty()) {
            continue;
        }
        
        std::string command, args;
        if (parseInput(input, command, args)) {
            // Handle command
            auto it = commandHandlers.find(command);
            if (it != commandHandlers.end()) {
                it->second(args);
            } else {
                std::cout << "Unknown command: " << command << std::endl;
                std::cout << "Type '.help' for a list of commands" << std::endl;
            }
        } else {
            // Process as SQL query
            executeQuery(input);
        }
    }
}

std::shared_ptr<Table> CommandLineInterface::processQuery(
    const std::string& query, QueryExecutor::ExecutionStats& stats) {
    
    std::string command, args;
    if (parseInput(query, command, args)) {
        // Handle command
        auto it = commandHandlers.find(command);
        if (it != commandHandlers.end()) {
            it->second(args);
            return nullptr;
        } else {
            resultDisplayer.displayError("Unknown command: " + command);
            return nullptr;
        }
    } else {
        // Process as SQL query
        try {
            return queryExecutor.executeQuery(query, stats);
        } catch (const std::exception& e) {
            resultDisplayer.displayError("Error executing query: " + std::string(e.what()));
            return nullptr;
        }
    }
}

void CommandLineInterface::setOutputMode(OutputMode mode) {
    outputMode = mode;
}

CommandLineInterface::OutputMode CommandLineInterface::getOutputMode() const {
    return outputMode;
}

void CommandLineInterface::initializeCommandHandlers() {
    commandHandlers[".help"] = [this](const std::string& args) { handleHelp(args); };
    commandHandlers[".exit"] = [this](const std::string& args) { handleExit(args); };
    commandHandlers[".quit"] = [this](const std::string& args) { handleExit(args); };
    commandHandlers[".tables"] = [this](const std::string& args) { handleListTables(args); };
    commandHandlers[".describe"] = [this](const std::string& args) { handleDescribeTable(args); };
    commandHandlers[".execute"] = [this](const std::string& args) { handleExecuteFile(args); };
    commandHandlers[".mode"] = [this](const std::string& args) { handleSetExecutionMode(args); };
    commandHandlers[".output"] = [this](const std::string& args) { handleSetOutputMode(args); };
    commandHandlers[".streaming"] = [this](const std::string& args) { handleToggleStreaming(args); };
}

void CommandLineInterface::handleHelp(const std::string& args) {
    std::cout << "Available commands:" << std::endl;
    std::cout << "  .help                   - Show this help message" << std::endl;
    std::cout << "  .exit, .quit            - Exit the program" << std::endl;
    std::cout << "  .tables                 - List all tables" << std::endl;
    std::cout << "  .describe <table_name>  - Show table schema" << std::endl;
    std::cout << "  .execute <file_path>    - Execute SQL queries from a file" << std::endl;
    std::cout << "  .mode <cpu|gpu|auto>    - Set execution mode" << std::endl;
    std::cout << "  .output <console|file|both> - Set output mode" << std::endl;
    std::cout << "  .streaming <on|off>     - Enable/disable CUDA streaming" << std::endl;
}

void CommandLineInterface::handleExit(const std::string& args) {
    std::cout << "Exiting..." << std::endl;
    running = false;
}

void CommandLineInterface::handleListTables(const std::string& args) {
    std::vector<std::string> tableNames = storageManager.getTableNames();
    
    if (tableNames.empty()) {
        std::cout << "No tables found" << std::endl;
        return;
    }
    
    std::cout << "Tables:" << std::endl;
    for (const auto& name : tableNames) {
        std::cout << "  " << name << std::endl;
    }
}

void CommandLineInterface::handleDescribeTable(const std::string& args) {
    std::string tableName = StringUtils::trim(args);
    
    if (tableName.empty()) {
        std::cout << "Usage: .describe <table_name>" << std::endl;
        return;
    }
    
    auto table = storageManager.getTable(tableName);
    if (!table) {
        std::cout << "Table not found: " << tableName << std::endl;
        return;
    }
    
    std::cout << "Table: " << tableName << std::endl;
    std::cout << "Columns:" << std::endl;
    
    const auto& columnNames = table->getColumnNames();
    for (size_t i = 0; i < columnNames.size(); ++i) {
        std::cout << "  " << columnNames[i];
        
        if (table->isColumnPrimaryKey(i)) {
            std::cout << " (Primary Key)";
        }
        
        if (table->isColumnForeignKey(i)) {
            std::string refTable = table->getForeignKeyReference(columnNames[i]);
            if (!refTable.empty()) {
                std::cout << " (Foreign Key -> " << refTable << ")";
            } else {
                std::cout << " (Foreign Key)";
            }
        }
        
        std::cout << std::endl;
    }
    
    std::cout << "Row count: " << table->getRowCount() << std::endl;
}

void CommandLineInterface::handleExecuteFile(const std::string& args) {
    std::string filePath = StringUtils::trim(args);
    
    if (filePath.empty()) {
        std::cout << "Usage: .execute <file_path>" << std::endl;
        return;
    }
    
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cout << "Failed to open file: " << filePath << std::endl;
        return;
    }
    
    std::string line;
    std::string currentQuery;
    int queryCount = 0;
    int successCount = 0;
    
    while (std::getline(file, line)) {
        std::string trimmedLine = StringUtils::trim(line);
        
        // Skip empty lines and comments
        if (trimmedLine.empty() || trimmedLine.starts_with("--")) {
            continue;
        }
        
        currentQuery += line + " ";
        
        // Check if query is complete (ends with semicolon)
        if (trimmedLine.ends_with(";")) {
            queryCount++;
            std::cout << "Executing query #" << queryCount << ": " << currentQuery << std::endl;
            
            try {
                QueryExecutor::ExecutionStats stats;
                auto result = queryExecutor.executeQuery(currentQuery, stats);
                
                if (result) {
                    resultDisplayer.displayTable(result);
                    resultDisplayer.displayStats(stats);
                    
                    if (outputMode == OutputMode::FILE_ONLY || outputMode == OutputMode::BOTH) {
                        std::string outputName = "result_" + std::to_string(queryCount);
                        resultDisplayer.saveResults(result, outputName);
                    }
                    
                    successCount++;
                }
            } catch (const std::exception& e) {
                std::cout << "Error executing query: " << e.what() << std::endl;
            }
            
            currentQuery.clear();
        }
    }
    
    std::cout << "Executed " << queryCount << " queries, " << successCount << " succeeded" << std::endl;
}

void CommandLineInterface::handleSetExecutionMode(const std::string& args) {
    std::string mode = StringUtils::toLower(StringUtils::trim(args));
    
    if (mode == "cpu") {
        queryExecutor.setExecutionMode(QueryExecutor::ExecutionMode::CPU);
        std::cout << "Execution mode set to CPU" << std::endl;
    } else if (mode == "gpu") {
        queryExecutor.setExecutionMode(QueryExecutor::ExecutionMode::GPU);
        std::cout << "Execution mode set to GPU" << std::endl;
    } else if (mode == "auto") {
        queryExecutor.setExecutionMode(QueryExecutor::ExecutionMode::AUTO);
        std::cout << "Execution mode set to AUTO" << std::endl;
    } else {
        std::cout << "Invalid mode. Use 'cpu', 'gpu', or 'auto'" << std::endl;
    }
}

void CommandLineInterface::handleSetOutputMode(const std::string& args) {
    std::string mode = StringUtils::toLower(StringUtils::trim(args));
    
    if (mode == "console") {
        setOutputMode(OutputMode::CONSOLE_ONLY);
        std::cout << "Output mode set to console only" << std::endl;
    } else if (mode == "file") {
        setOutputMode(OutputMode::FILE_ONLY);
        std::cout << "Output mode set to file only" << std::endl;
    } else if (mode == "both") {
        setOutputMode(OutputMode::BOTH);
        std::cout << "Output mode set to both console and file" << std::endl;
    } else {
        std::cout << "Invalid mode. Use 'console', 'file', or 'both'" << std::endl;
    }
}

void CommandLineInterface::handleToggleStreaming(const std::string& args) {
    std::string mode = StringUtils::toLower(StringUtils::trim(args));
    
    if (mode == "on") {
        queryExecutor.enableStreaming(true);
        std::cout << "CUDA streaming enabled" << std::endl;
    } else if (mode == "off") {
        queryExecutor.enableStreaming(false);
        std::cout << "CUDA streaming disabled" << std::endl;
    } else {
        std::cout << "Invalid option. Use 'on' or 'off'" << std::endl;
    }
}

void CommandLineInterface::executeQuery(const std::string& query) {
    try {
        QueryExecutor::ExecutionStats stats;
        auto result = queryExecutor.executeQuery(query, stats);
        
        if (result) {
            // Display result based on output mode
            if (outputMode == OutputMode::CONSOLE_ONLY || outputMode == OutputMode::BOTH) {
                resultDisplayer.displayTable(result);
                resultDisplayer.displayStats(stats);
            }
            
            // Save to file if needed
            if (outputMode == OutputMode::FILE_ONLY || outputMode == OutputMode::BOTH) {
                std::string outputName = "result_" + std::to_string(std::time(nullptr));
                resultDisplayer.saveResults(result, outputName);
                
                if (outputMode == OutputMode::FILE_ONLY) {
                    std::cout << "Result saved to file: " << outputName << std::endl;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cout << "Error executing query: " << e.what() << std::endl;
    }
}

std::string CommandLineInterface::getInput() {
    std::cout << "SQL> ";
    std::string input;
    std::getline(std::cin, input);
    return input;
}

bool CommandLineInterface::parseInput(const std::string& input, std::string& command, std::string& args) {
    std::string trimmedInput = StringUtils::trim(input);
    
    // Check if input is a command (starts with '.')
    if (trimmedInput.empty() || trimmedInput[0] != '.') {
        return false;
    }
    
    // Find the first space to separate command and arguments
    size_t spacePos = trimmedInput.find(' ');
    if (spacePos == std::string::npos) {
        // No arguments
        command = trimmedInput;
        args = "";
    } else {
        command = trimmedInput.substr(0, spacePos);
        args = StringUtils::trim(trimmedInput.substr(spacePos + 1));
    }
    
    return true;
}

} // namespace SQLQueryProcessor