#include "../../include/SQLProcessing/SQLQueryProcessor.hpp"
#include "hsql/SQLParser.h"
#include "hsql/SQLParserResult.h"
#include "hsql/sql/Expr.h"
#include "hsql/sql/SelectStatement.h"
#include "hsql/sql/CreateStatement.h"
#include "hsql/sql/InsertStatement.h"
#include "../../include/Operations/Select.hpp"
#include "../../include/Operations/Project.hpp"
#include "../../include/Operations/Join.hpp"
#include <iostream>
#include <stdexcept>

namespace GPUDBMS
{

    SQLQueryProcessor::SQLQueryProcessor()
    {
        // Don't initialize with hardcoded tables
    }

    // Add tables to the processor
    void SQLQueryProcessor::registerTable(const std::string& name, const Table& table) {
        tables[name] = table;
    }

    // Create table from schema definition
    Table SQLQueryProcessor::createTableFromSchema(const hsql::CreateStatement* stmt) {
        std::vector<Column> columns;
        
        // Process column definitions
        for (hsql::ColumnDefinition* col : *stmt->columns) {
            DataType dataType;
            
            // Map SQL types to your DataType enum
            switch (col->type.data_type) {
                case hsql::DataType::INT:
                    dataType = DataType::INT;
                    break;
                case hsql::DataType::DOUBLE:
                    dataType = DataType::DOUBLE;
                    break;
                case hsql::DataType::TEXT:
                case hsql::DataType::VARCHAR:
                    dataType = DataType::VARCHAR;
                    break;
                default:
                    throw std::runtime_error("Unsupported data type for column: " + std::string(col->name));
            }
            
            columns.push_back(Column(col->name, dataType));
        }
        
        // Create and return the table
        return Table(columns);
    }

    SQLQueryProcessor::~SQLQueryProcessor()
    {
        // Cleanup if needed
    }

    Table SQLQueryProcessor::processQuery(const std::string &query)
    {
        // Parse the SQL query
        hsql::SQLParserResult result;
        hsql::SQLParser::parse(query, &result);

        if (!result.isValid())
        {
            throw std::runtime_error("Failed to parse SQL query: " +
                                     (result.errorMsg() ? std::string(result.errorMsg()) : "Unknown error"));
        }

        // Process each statement (we'll just handle the first one for now)
        if (result.size() > 0)
        {
            const hsql::SQLStatement *stmt = result.getStatement(0);

            switch (stmt->type())
            {
            case hsql::kStmtSelect:
                return executeSelectStatement(static_cast<const hsql::SelectStatement *>(stmt));
            case hsql::kStmtCreate:
                return executeCreateStatement(static_cast<const hsql::CreateStatement *>(stmt));
            case hsql::kStmtInsert:
                return executeInsertStatement(static_cast<const hsql::InsertStatement *>(stmt));
            default:
                throw std::runtime_error("Unsupported SQL statement type");
            }
        }

        throw std::runtime_error("No SQL statements found");
    }

    // Implementation for CREATE TABLE
    Table SQLQueryProcessor::executeCreateStatement(const hsql::CreateStatement* stmt) {
        if (stmt->type != hsql::kCreateTable) {
            throw std::runtime_error("Only CREATE TABLE is supported");
        }
        
        // Create table from schema
        Table newTable = createTableFromSchema(stmt);
        
        // Register the table
        tables[stmt->tableName] = newTable;
        
        // Return the empty table
        return newTable;
    }
    
    // Implementation for INSERT INTO
    Table SQLQueryProcessor::executeInsertStatement(const hsql::InsertStatement* stmt) {
        // Get the target table
        auto it = tables.find(stmt->tableName);
        if (it == tables.end()) {
            throw std::runtime_error("Table not found: " + std::string(stmt->tableName));
        }
        
        Table& table = it->second;
        
        // Handle VALUES clause
        if (stmt->type == hsql::kInsertValues) {
            for (const hsql::Expr* valueList : *stmt->values) {
                // Get references to column data
                std::vector<std::reference_wrapper<ColumnData>> columnData;
                
                // If columns are specified, use them; otherwise use all columns in order
                std::vector<std::string> columnNames;
                if (stmt->columns) {
                    for (const char* colName : *stmt->columns) {
                        columnNames.push_back(colName);
                    }
                } else {
                    // Use all columns
                    for (size_t i = 0; i < table.getColumnCount(); i++) {
                        columnNames.push_back(table.getColumnName(i));
                    }
                }
                
                // Insert values into respective columns
                for (size_t i = 0; i < columnNames.size(); i++) {
                    const hsql::Expr* expr = valueList->exprList->at(i);
                    const std::string& colName = columnNames[i];
                    
                    // Get column data type
                    auto colType = table.getColumnType(colName);
                    
                    // Insert based on data type
                    switch (colType) {
                        case DataType::INT: {
                            if (expr->type != hsql::kExprLiteralInt) {
                                throw std::runtime_error("Type mismatch for column " + colName);
                            }
                            auto& col = static_cast<ColumnDataImpl<int>&>(table.getColumnData(colName));
                            col.append(expr->ival);
                            break;
                        }
                        case DataType::DOUBLE: {
                            double value;
                            if (expr->type == hsql::kExprLiteralFloat) {
                                value = expr->fval;
                            } else if (expr->type == hsql::kExprLiteralInt) {
                                value = static_cast<double>(expr->ival);
                            } else {
                                throw std::runtime_error("Type mismatch for column " + colName);
                            }
                            auto& col = static_cast<ColumnDataImpl<double>&>(table.getColumnData(colName));
                            col.append(value);
                            break;
                        }
                        case DataType::VARCHAR: {
                            if (expr->type != hsql::kExprLiteralString) {
                                throw std::runtime_error("Type mismatch for column " + colName);
                            }
                            auto& col = static_cast<ColumnDataImpl<std::string>&>(table.getColumnData(colName));
                            col.append(expr->name);
                            break;
                        }
                        default:
                            throw std::runtime_error("Unsupported data type for column " + colName);
                    }
                }
                
                // Finalize the row
                table.finalizeRow();
            }
        } else {
            throw std::runtime_error("Only INSERT INTO VALUES is supported");
        }
        
        return table;
    }

    Table &SQLQueryProcessor::getTable(const std::string &name)
    {
        auto it = tables.find(name);
        if (it == tables.end())
        {
            throw std::runtime_error("Table not found: " + name);
        }
        return it->second;
    }

    Table SQLQueryProcessor::executeSelectStatement(const hsql::SelectStatement* stmt) {
        // Basic implementation to demonstrate integration
        
        // Handle FROM clause (can be a table or a join)
        if (!stmt->fromTable) {
            throw std::runtime_error("FROM clause is required");
        }
        
        Table resultTable;
        
        // Check if we're dealing with a table or a join
        if (stmt->fromTable->type == hsql::kTableName) {
            // Simple table reference
            resultTable = getTable(stmt->fromTable->name);
        } 
        else if (stmt->fromTable->type == hsql::kTableJoin) {
            // Handle JOIN
            const hsql::JoinDefinition* join = stmt->fromTable->join;
            
            // Get the left and right tables
            Table leftTable = getTable(join->left->name);
            Table rightTable = getTable(join->right->name);
            
            // Translate the join condition
            auto joinCondition = translateWhereCondition(join->condition);
            
            // Determine join type
            JoinType joinType;
            switch (join->type) {
                case hsql::kJoinInner:
                    joinType = JoinType::INNER;
                    break;
                case hsql::kJoinLeft:
                    joinType = JoinType::LEFT;
                    break;
                case hsql::kJoinRight:
                    joinType = JoinType::RIGHT;
                    break;
                default:
                    throw std::runtime_error("Unsupported join type");
            }
            
            // Execute the join
            Join joinOp(leftTable, rightTable, *joinCondition, joinType);
            resultTable = joinOp.executeCPU();
        } 
        else {
            throw std::runtime_error("Unsupported FROM clause type");
        }
        
        // Apply WHERE condition if it exists
        if (stmt->whereClause) {
            auto condition = translateWhereCondition(stmt->whereClause);
            Select selectOp(resultTable, *condition);
            resultTable = selectOp.executeCPU();
        }
        
        // Handle SELECT columns (projection)
        // Handle SELECT columns (projection)
if (!stmt->selectList->empty() && (*stmt->selectList)[0]->type != hsql::kExprStar) {
    std::vector<std::string> projectColumns;
    for (const hsql::Expr* expr : *stmt->selectList) {
        if (expr->type == hsql::kExprColumnRef) {
            projectColumns.push_back(expr->name);
        }
    }
    
    if (!projectColumns.empty()) {
        Project projectOp(resultTable, projectColumns);
        resultTable = projectOp.executeCPU(); // This line is missing assignment to resultTable
    }
}
        
        return resultTable;
    }
    
    // You may also need to implement the translateWhereCondition method
    std::unique_ptr<Condition> SQLQueryProcessor::translateWhereCondition(const hsql::Expr* expr) {
        if (!expr) return nullptr;
        
        switch (expr->type) {
            case hsql::kExprOperator: {
                // Handle binary operators
                if (expr->expr && expr->expr2) {
                    // Get the column name (left side of condition)
                    std::string columnName;
                    if (expr->expr->type == hsql::kExprColumnRef) {
                        // Handle table aliases if present (e.g., e.id)
                        if (expr->expr->table) {
                            columnName = std::string(expr->expr->table) + "." + expr->expr->name;
                        } else {
                            columnName = expr->expr->name;
                        }
                    } else {
                        throw std::runtime_error("Left side of condition must be a column reference");
                    }
                    
                    // Handle based on operator type
                    switch (expr->opType) {
                        case hsql::kOpEquals: {
                            // Handle different types of right operands
                            if (expr->expr2->type == hsql::kExprLiteralString) {
                                return ConditionBuilder::equals(columnName, expr->expr2->name);
                            } else if (expr->expr2->type == hsql::kExprLiteralInt) {
                                return ConditionBuilder::equals(columnName, std::to_string(expr->expr2->ival));
                            } else if (expr->expr2->type == hsql::kExprLiteralFloat) {
                                return ConditionBuilder::equals(columnName, std::to_string(expr->expr2->fval));
                            }
                            break;
                        }
                        case hsql::kOpGreater: {
                            if (expr->expr2->type == hsql::kExprLiteralInt) {
                                return ConditionBuilder::greaterThan(columnName, std::to_string(expr->expr2->ival));
                            } else if (expr->expr2->type == hsql::kExprLiteralFloat) {
                                return ConditionBuilder::greaterThan(columnName, std::to_string(expr->expr2->fval));
                            }
                            break;
                        }
                        case hsql::kOpLess: {
                            if (expr->expr2->type == hsql::kExprLiteralInt) {
                                return ConditionBuilder::lessThan(columnName, std::to_string(expr->expr2->ival));
                            } else if (expr->expr2->type == hsql::kExprLiteralFloat) {
                                return ConditionBuilder::lessThan(columnName, std::to_string(expr->expr2->fval));
                            }
                            break;
                        }
                        case hsql::kOpAnd: {
                            auto leftCond = translateWhereCondition(expr->expr);
                            auto rightCond = translateWhereCondition(expr->expr2);
                            return ConditionBuilder::And(std::move(leftCond), std::move(rightCond));
                        }
                        case hsql::kOpOr: {
                            auto leftCond = translateWhereCondition(expr->expr);
                            auto rightCond = translateWhereCondition(expr->expr2);
                            return ConditionBuilder::Or(std::move(leftCond), std::move(rightCond));
                        }
                        default:
                            throw std::runtime_error("Unsupported operator in WHERE clause");
                    }
                }
                break;
            }
            default:
                break;
        }
        
        throw std::runtime_error("Unsupported expression type in WHERE clause");
    }
}