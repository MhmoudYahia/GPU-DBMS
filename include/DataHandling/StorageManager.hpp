#pragma once
#include "Table.hpp"
#include "CSVProcessor.hpp"
#include <string>
#include <unordered_map>
#include <memory>
#include <filesystem>
#include <vector>

namespace SQLQueryProcessor
{

    class StorageManager
    {
    private:
        std::unordered_map<std::string, std::shared_ptr<Table>> tables;
        std::string dataDirectory;
        std::string outputCSVDirectory;
        std::string outputTXTDirectory;
        CSVProcessor csvProcessor;

    public:
        explicit StorageManager(const std::string &dataDir,
                                const std::string &outputCSVDir = "data/outputs/csv/",
                                const std::string &outputTXTDir = "data/outputs/txt/");
        ~StorageManager() = default;

        // Load tables from CSV files in the data directory
        void loadTables();

        // Load a specific table from a CSV file
        std::shared_ptr<Table> loadTable(const std::string &filePath);

        // Get a specific table by name
        std::shared_ptr<Table> getTable(const std::string &tableName) const;

        // List all available tables
        std::vector<std::string> getTableNames() const;

        // Save result table to CSV
        bool saveResultCSV(const std::shared_ptr<Table> &resultTable, const std::string &outputName);

        // Save result to text file
        bool saveResultTXT(const std::string &result, const std::string &outputName);

        // Check if a table exists
        bool tableExists(const std::string &tableName) const;
    };

}