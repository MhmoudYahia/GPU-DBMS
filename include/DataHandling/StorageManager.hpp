#pragma once

#include "Table.hpp"
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>

namespace GPUDBMS {

struct ForeignKeyConstraint {
    std::string sourceColumn;
    std::string targetTable;
    std::string targetColumn;
};

struct TableMetadata {
    std::string name;
    std::vector<std::string> primaryKeys;
    std::vector<ForeignKeyConstraint> foreignKeys;
    std::string filePath;
};

class StorageManager {
public:
    StorageManager(const std::string& dataDirectory);
    ~StorageManager();

    // Load all tables from data directory
    void loadAllTables();

        // Get the data directory
        const std::string& getDataDirectory() const {
            return m_dataDirectory;
        }
    
    // Load a specific table
    Table loadTableFromCSV(const std::string& tableName);
    
    // Save a table to CSV
    void saveTableToCSV(const std::string& tableName, const Table& table);
    
    // Get loaded table
    Table& getTable(const std::string& tableName);
    
    // Check if table exists
    bool hasTable(const std::string& tableName) const;
    
    // Register a new table
    void registerTable(const std::string& tableName, const Table& table);
    
    // Get table names
    std::vector<std::string> getTableNames() const;
    
    // Get metadata for a table
    const TableMetadata& getTableMetadata(const std::string& tableName) const;

private:
    std::string m_dataDirectory;
    std::unordered_map<std::string, Table> m_tables;
    std::unordered_map<std::string, TableMetadata> m_tableMetadata;
    
    // Parse CSV file header to get schema information
    Table createTableFromCSVHeader(const std::string& filePath);
    
    // Detect primary and foreign keys from headers
    void parseTableConstraints(const std::string& tableName, const std::vector<std::string>& headers);
};

} // namespace GPUDBMS