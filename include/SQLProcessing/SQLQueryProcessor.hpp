#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include "../DataHandling/Table.hpp"
#include "../DataHandling/Condition.hpp"
#include "../DataHandling/StorageManager.hpp"

// Forward declarations for SQL parser types
namespace hsql
{
    class SQLStatement;
    class SelectStatement;
    class CreateStatement;
    class InsertStatement;
    class Expr;
}

namespace GPUDBMS
{

    class SQLQueryProcessor
    {
    public:
        SQLQueryProcessor();
        SQLQueryProcessor(const std::string &dataDirectory);
        ~SQLQueryProcessor();

        // Process a SQL query and return the result table
        Table processQuery(const std::string& query, bool useGPU = false);

        /**
         * @brief Get a list of all available table names
         * @return Vector of table names
         */
        std::vector<std::string> getTableNames() const;

        // Add tables to the processor
        void registerTable(const std::string &name, const Table &table);

        // Get a reference to a table
        Table &getTable(const std::string &name);

        // Load a table from CSV
        Table loadTableFromCSV(const std::string &tableName);

        // Save a table to CSV
        void saveTableToCSV(const std::string &tableName, const Table &table);

        // Save query result to CSV file with a specific filename
        void saveQueryResultToCSV(const Table &resultTable, const std::string &filename);

        // Process query and save the result in one step
        Table processQueryAndSave(const std::string &query, const std::string &outputFilename);

    private:
        // Tables in memory - in a real implementation, you'd have a proper catalog
        std::unordered_map<std::string, Table> tables;
        std::unique_ptr<StorageManager> storageManager;

        // Helper methods to handle different statement types
        Table executeSelectStatement(const hsql::SelectStatement* stmt, bool useGPU = false);
        Table executeCreateStatement(const hsql::CreateStatement *stmt);
        Table executeInsertStatement(const hsql::InsertStatement *stmt);

        // Helper to create a table from schema definition
        Table createTableFromSchema(const hsql::CreateStatement *stmt);

        // Helper to convert SQL expressions to conditions
        std::unique_ptr<Condition> translateWhereCondition(const hsql::Expr *expr);
    };

} // namespace GPUDBMS