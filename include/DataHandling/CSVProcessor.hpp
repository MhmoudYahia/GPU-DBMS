#pragma once
#include "Table.hpp"
#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>

namespace SQLQueryProcessor
{

    enum class ColumnType
    {
        TEXT,
        NUMERIC,
        DATETIME,
        UNKNOWN
    };

    class CSVProcessor
    {
    public:
        CSVProcessor() = default;
        ~CSVProcessor() = default;

        // Read a CSV file and convert it to a Table
        std::shared_ptr<Table> readCSV(const std::string &filePath);

        // Write a Table to a CSV file
        bool writeCSV(const std::shared_ptr<Table> &table, const std::string &filePath);

    private:
        // Helper methods for parsing
        std::vector<std::string> parseHeader(const std::string &headerLine);
        std::vector<std::string> parseLine(const std::string &line);

        // Parse column metadata from header
        struct ColumnMeta
        {
            std::string name;
            ColumnType type;
            bool isPrimary;
            bool isForeign;
            std::string referencedTable;
        };

        ColumnMeta parseColumnHeader(const std::string &header);

        // Check if a column name indicates a foreign key reference
        bool isForeignKeyReference(const std::string &columnName, std::string &referencedTable);
    };

}