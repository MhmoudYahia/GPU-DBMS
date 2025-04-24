#include "DataHandling/StorageManager.hpp"
#include "Utilities/Logger.hpp"
#include "Utilities/ErrorHandling.hpp"
#include <filesystem>
#include <fstream>
#include <algorithm>

namespace SQLQueryProcessor {

StorageManager::StorageManager(const std::string& dataDir, 
                             const std::string& outputCsvDir,
                             const std::string& outputTxtDir)
    : dataDirectory(dataDir), 
      outputCSVDirectory(outputCsvDir), 
      outputTXTDirectory(outputTxtDir) {
    
    Logger::debug("Storage manager initialized with data directory: " + dataDir);
    
    // Ensure directories exist
    std::filesystem::create_directories(dataDirectory);
    std::filesystem::create_directories(outputCSVDirectory);
    std::filesystem::create_directories(outputTXTDirectory);
}

void StorageManager::loadTables() {
    Logger::info("Loading tables from directory: " + dataDirectory);
    
    tables.clear();
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(dataDirectory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".csv") {
                std::string filePath = entry.path().string();
                loadTable(filePath);
            }
        }
        
        Logger::info("Loaded " + std::to_string(tables.size()) + " tables");
    } catch (const std::filesystem::filesystem_error& e) {
        throw DataException("Error accessing data directory: " + std::string(e.what()));
    }
}

std::shared_ptr<Table> StorageManager::loadTable(const std::string& filePath) {
    Logger::debug("Loading table from file: " + filePath);
    
    try {
        std::shared_ptr<Table> table = csvProcessor.readCSV(filePath);
        tables[table->getName()] = table;
        Logger::info("Loaded table: " + table->getName());
        return table;
    } catch (const std::exception& e) {
        Logger::error("Failed to load table from " + filePath + ": " + e.what());
        throw;
    }
}

std::shared_ptr<Table> StorageManager::getTable(const std::string& tableName) const {
    auto it = tables.find(tableName);
    if (it == tables.end()) {
        return nullptr;
    }
    return it->second;
}

std::vector<std::string> StorageManager::getTableNames() const {
    std::vector<std::string> names;
    names.reserve(tables.size());
    
    for (const auto& pair : tables) {
        names.push_back(pair.first);
    }
    
    // Sort names alphabetically
    std::sort(names.begin(), names.end());
    
    return names;
}

bool StorageManager::saveResultCSV(const std::shared_ptr<Table>& resultTable, const std::string& outputName) {
    if (!resultTable) {
        Logger::error("Cannot save null result table");
        return false;
    }
    
    std::string filePath = outputCSVDirectory + outputName;
    if (!filePath.ends_with(".csv")) {
        filePath += ".csv";
    }
    
    Logger::info("Saving result to CSV: " + filePath);
    return csvProcessor.writeCSV(resultTable, filePath);
}

bool StorageManager::saveResultTXT(const std::string& result, const std::string& outputName) {
    std::string filePath = outputTXTDirectory + outputName;
    if (!filePath.ends_with(".txt")) {
        filePath += ".txt";
    }
    
    Logger::info("Saving result to text file: " + filePath);
    
    try {
        std::ofstream file(filePath);
        if (!file.is_open()) {
            Logger::error("Failed to open file for writing: " + filePath);
            return false;
        }
        
        file << result;
        return true;
    } catch (const std::exception& e) {
        Logger::error("Error writing to file " + filePath + ": " + e.what());
        return false;
    }
}

bool StorageManager::tableExists(const std::string& tableName) const {
    return tables.find(tableName) != tables.end();
}

} // namespace SQLQueryProcessor