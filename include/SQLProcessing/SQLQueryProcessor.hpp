#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include "../DataHandling/Table.hpp"
#include "../DataHandling/Condition.hpp"

// Forward declarations for SQL parser types
namespace hsql {
    class SQLStatement;
    class SelectStatement;
    class CreateStatement;
    class InsertStatement;
    class Expr;
}

namespace GPUDBMS {

class SQLQueryProcessor {
public:
    SQLQueryProcessor();
    ~SQLQueryProcessor();

    // Process a SQL query and return the result table
    Table processQuery(const std::string& query);
    
    // Add tables to the processor
    void registerTable(const std::string& name, const Table& table);

private:
    // Tables in memory - in a real implementation, you'd have a proper catalog
    std::unordered_map<std::string, Table> tables;
    
    // Helper methods to handle different statement types
    Table executeSelectStatement(const hsql::SelectStatement* stmt);
    Table executeCreateStatement(const hsql::CreateStatement* stmt);  // Add this declaration
    Table executeInsertStatement(const hsql::InsertStatement* stmt);  // Add this declaration
    
    // Helper to create a table from schema definition
    Table createTableFromSchema(const hsql::CreateStatement* stmt);
    
    // Helper to convert SQL expressions to conditions
    std::unique_ptr<Condition> translateWhereCondition(const hsql::Expr* expr);
    
    // Helper to get an existing table by name
    Table& getTable(const std::string& name);
};

} // namespace GPUDBMS