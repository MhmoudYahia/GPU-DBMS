#include "../../include/DataHandling/StorageManager.hpp"
#include "../../include/DataHandling/CSVProcessor.hpp"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <regex>
#include <stdexcept>

namespace GPUDBMS {

StorageManager::StorageManager(const std::string& dataDirectory)
    : m_dataDirectory(dataDirectory)
{
    // Ensure the data directory exists
    std::filesystem::create_directories(dataDirectory);
    std::filesystem::create_directories(dataDirectory + "/outputs/csv");
    std::filesystem::create_directories(dataDirectory + "/outputs/txt");
}

StorageManager::~StorageManager() {
    // Save all modified tables before destruction
    for (const auto& [tableName, table] : m_tables) {
        // This could be optimized to only save modified tables
        saveTableToCSV(tableName, table);
    }
}

void StorageManager::loadAllTables() {
    const std::string inputDir = m_dataDirectory + "/input_csvs";
    
    if (!std::filesystem::exists(inputDir)) {
        std::cerr << "Warning: Input directory does not exist: " << inputDir << std::endl;
        return;
    }

    for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".csv") {
            std::string tableName = entry.path().stem().string();
            try {
                Table table = loadTableFromCSV(tableName);
                m_tables[tableName] = table;
                
                // Set metadata
                TableMetadata metadata;
                metadata.name = tableName;
                metadata.filePath = entry.path().string();
                m_tableMetadata[tableName] = metadata;
                
                std::cout << "Loaded table: " << tableName << " with " 
                          << table.getRowCount() << " rows and " 
                          << table.getColumnCount() << " columns" << std::endl;
            }
            catch (const std::exception& e) {
                std::cerr << "Error loading table " << tableName << ": " << e.what() << std::endl;
            }
        }
    }
    
    // After loading all tables, parse constraints to establish relationships
    for (const auto& [tableName, metadata] : m_tableMetadata) {
        const Table& table = m_tables[tableName];
        std::vector<std::string> headers;
        for (size_t i = 0; i < table.getColumnCount(); ++i) {
            headers.push_back(table.getColumnName(i));
        }
        parseTableConstraints(tableName, headers);
    }
}

Table StorageManager::loadTableFromCSV(const std::string& tableName) {
    const std::string filePath = m_dataDirectory + "/input_csvs/" + tableName + ".csv";
    
    if (!std::filesystem::exists(filePath)) {
        throw std::runtime_error("CSV file not found: " + filePath);
    }
    
    // Use CSVProcessor to load the file
    CSVProcessor processor;
    Table table = processor.readCSV(filePath);
    
    // Store the table in our cache
    m_tables[tableName] = table;
    
    // Setup metadata
    TableMetadata metadata;
    metadata.name = tableName;
    metadata.filePath = filePath;
    m_tableMetadata[tableName] = metadata;
    
    // Parse constraints
    std::vector<std::string> headers;
    for (size_t i = 0; i < table.getColumnCount(); ++i) {
        headers.push_back(table.getColumnName(i));
    }
    parseTableConstraints(tableName, headers);
    
    return table;
}

void StorageManager::saveTableToCSV(const std::string& tableName, const Table& table) {
    // Default to saving in the outputs directory
    const std::string outputPath = m_dataDirectory + "/outputs/csv/" + tableName + ".csv";
    
    CSVProcessor processor;
    processor.writeCSV(table, outputPath);
    
    std::cout << "Saved table " << tableName << " to " << outputPath << std::endl;
}

void StorageManager::parseTableConstraints(const std::string& tableName, const std::vector<std::string>& headers) {
    // Regular expressions for detecting primary and foreign keys
    std::regex primaryKeyRegex("(.+)\\s*\\(P\\)"); // Column(P) format
    std::regex foreignKeyRegex("#(.+)_(.+)");      // #TableName_columnName format
    
    TableMetadata& metadata = m_tableMetadata[tableName];
    
    for (const auto& header : headers) {
        std::smatch match;
        
        // Check if it's a primary key
        if (std::regex_match(header, match, primaryKeyRegex)) {
            std::string columnName = match[1].str();
            // Remove trailing spaces
            columnName.erase(columnName.find_last_not_of(" ") + 1);
            metadata.primaryKeys.push_back(columnName);
        }
        
        // Check if it's a foreign key
        else if (std::regex_match(header, match, foreignKeyRegex)) {
            std::string targetTable = match[1].str();
            std::string targetColumn = match[2].str();
            
            ForeignKeyConstraint fk;
            fk.sourceColumn = header;
            fk.targetTable = targetTable;
            fk.targetColumn = targetColumn;
            
            metadata.foreignKeys.push_back(fk);
        }
    }
}

Table& StorageManager::getTable(const std::string& tableName) {
    if (!hasTable(tableName)) {
        // Try to load it first
        try {
            loadTableFromCSV(tableName);
        } catch (const std::exception& e) {
            throw std::runtime_error("Table not found: " + tableName);
        }
    }
    
    return m_tables.at(tableName);
}

bool StorageManager::hasTable(const std::string& tableName) const {
    return m_tables.find(tableName) != m_tables.end();
}

void StorageManager::registerTable(const std::string& tableName, const Table& table) {
    m_tables[tableName] = table;
    
    // Setup basic metadata
    TableMetadata metadata;
    metadata.name = tableName;
    metadata.filePath = m_dataDirectory + "/outputs/csv/" + tableName + ".csv";
    m_tableMetadata[tableName] = metadata;
}

std::vector<std::string> StorageManager::getTableNames() const {
    std::vector<std::string> names;
    names.reserve(m_tables.size());
    
    for (const auto& [name, _] : m_tables) {
        names.push_back(name);
    }
    
    return names;
}

const TableMetadata& StorageManager::getTableMetadata(const std::string& tableName) const {
    auto it = m_tableMetadata.find(tableName);
    if (it == m_tableMetadata.end()) {
        throw std::runtime_error("No metadata found for table: " + tableName);
    }
    return it->second;
}

} // namespace GPUDBMS