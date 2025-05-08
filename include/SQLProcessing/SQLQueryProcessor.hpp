#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include "../DataHandling/Table.hpp"
#include "../DataHandling/Condition.hpp"
#include "../DataHandling/CSVProcessor.hpp"
#include "../../include/DataHandling/StorageManager.hpp"

// Forward declarations for SQL parser
namespace hsql
{
class SelectStatement;
class CreateStatement;
class InsertStatement;
class Expr;
} // namespace hsql

namespace GPUDBMS
{

class SQLQueryProcessor
{
public:
    SQLQueryProcessor();
    SQLQueryProcessor(const std::string &dataDirectory);
    ~SQLQueryProcessor();

    // Process an SQL query and return the result
    Table processQuery(const std::string &query, bool useGPU = false);

    // Process a query and save the result to a CSV file
    Table processQueryAndSave(const std::string &query, const std::string &outputFilename);

    // Load a table from CSV
    Table loadTableFromCSV(const std::string &tableName);

    // Save a table to CSV
    void saveTableToCSV(const std::string &tableName, const Table &table);

    // Save a query result to CSV
    void saveQueryResultToCSV(const Table &resultTable, const std::string &filename);

    // Get a reference to a table by name
    Table &getTable(const std::string &name);

    // Register a table with the processor
    void registerTable(const std::string &name, const Table &table);

    // Get all available table names
    std::vector<std::string> getTableNames() const;
    

private:
    // Map of table names to tables
    std::unordered_map<std::string, Table> tables;

    // Storage manager for loading/saving tables
    std::unique_ptr<StorageManager> storageManager;

    // Execute different types of SQL statements
    Table executeSelectStatement(const hsql::SelectStatement *stmt, bool useGPU = false);
    Table executeCreateStatement(const hsql::CreateStatement *stmt);
    Table executeInsertStatement(const hsql::InsertStatement *stmt);

    // Create a table from schema definition
    Table createTableFromSchema(const hsql::CreateStatement *stmt);

    // Translate WHERE condition to internal representation
    std::unique_ptr<Condition> translateWhereCondition(const hsql::Expr *expr);
    
    // New overload that supports table aliases
    std::unique_ptr<Condition> translateWhereCondition(
        const hsql::Expr *expr, 
        const std::unordered_map<std::string, std::string>& tableAliases);
};

} // namespace GPUDBMS