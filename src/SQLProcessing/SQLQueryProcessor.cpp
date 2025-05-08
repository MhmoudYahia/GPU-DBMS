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
#include "../../include/Operations/OrderBy.hpp"
#include "../../include/Operations/Aggregator.hpp"
#include "../../include/DataHandling/CSVProcessor.hpp"
#include <iostream>
#include <stdexcept>

namespace GPUDBMS
{

    SQLQueryProcessor::SQLQueryProcessor()
    {
    }

    SQLQueryProcessor::SQLQueryProcessor(const std::string &dataDirectory)
    {
        storageManager = std::make_unique<StorageManager>(dataDirectory);
        storageManager->loadAllTables();

        // Make all tables available to the query processor
        for (const auto &tableName : storageManager->getTableNames())
        {
            tables[tableName] = storageManager->getTable(tableName);
        }
    }

    Table SQLQueryProcessor::loadTableFromCSV(const std::string &tableName)
    {
        if (!storageManager)
        {
            throw std::runtime_error("StorageManager not initialized - cannot load table");
        }

        Table table = storageManager->loadTableFromCSV(tableName);
        tables[tableName] = table;
        return table;
    }

    void SQLQueryProcessor::saveTableToCSV(const std::string &tableName, const Table &table)
    {
        if (!storageManager)
        {
            throw std::runtime_error("StorageManager not initialized - cannot save table");
        }

        storageManager->saveTableToCSV(tableName, table);
    }

    void SQLQueryProcessor::saveQueryResultToCSV(const Table &resultTable, const std::string &filename)
    {
        if (!storageManager)
        {
            throw std::runtime_error("StorageManager not initialized - cannot save query result");
        }

        CSVProcessor csvProcessor;
        std::string outputPath = storageManager->getDataDirectory() + "/outputs/csv/" + filename + ".csv";
        csvProcessor.writeCSV(resultTable, outputPath);
        std::cout << "Saved query result to " << outputPath << std::endl;
    }

    // Process query and save the result in one step
    Table SQLQueryProcessor::processQueryAndSave(const std::string &query, const std::string &outputFilename)
    {
        Table result = processQuery(query);
        saveQueryResultToCSV(result, outputFilename);
        return result;
    }
    // Add tables to the processor
    void SQLQueryProcessor::registerTable(const std::string &name, const Table &table)
    {
        tables[name] = table;
    }

    // Create table from schema definition
    Table SQLQueryProcessor::createTableFromSchema(const hsql::CreateStatement *stmt)
    {
        std::vector<Column> columns;

        // Process column definitions
        for (hsql::ColumnDefinition *col : *stmt->columns)
        {
            DataType dataType;

            // Map SQL types to your DataType enum
            switch (col->type.data_type)
            {
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
            case hsql::DataType::FLOAT:
                dataType = DataType::FLOAT;
                break;
            case hsql::DataType::BOOLEAN:
                dataType = DataType::BOOL;
                break;
            case hsql::DataType::DATE:
                dataType = DataType::DATE;
                break;
            case hsql::DataType::DATETIME:
                dataType = DataType::DATETIME;
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

    Table SQLQueryProcessor::processQuery(const std::string &query, bool useGPU)
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
                return executeSelectStatement(static_cast<const hsql::SelectStatement *>(stmt), useGPU);
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
    Table SQLQueryProcessor::executeCreateStatement(const hsql::CreateStatement *stmt)
    {
        if (stmt->type != hsql::kCreateTable)
        {
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
    Table SQLQueryProcessor::executeInsertStatement(const hsql::InsertStatement *stmt)
    {
        // Get the target table
        auto it = tables.find(stmt->tableName);
        if (it == tables.end())
        {
            throw std::runtime_error("Table not found: " + std::string(stmt->tableName));
        }

        Table &table = it->second;

        // Handle VALUES clause
        if (stmt->type == hsql::kInsertValues)
        {
            for (const hsql::Expr *valueList : *stmt->values)
            {
                // Get references to column data
                std::vector<std::reference_wrapper<ColumnData>> columnData;

                // If columns are specified, use them; otherwise use all columns in order
                std::vector<std::string> columnNames;
                if (stmt->columns)
                {
                    for (const char *colName : *stmt->columns)
                    {
                        columnNames.push_back(colName);
                    }
                }
                else
                {
                    // Use all columns
                    for (size_t i = 0; i < table.getColumnCount(); i++)
                    {
                        columnNames.push_back(table.getColumnName(i));
                    }
                }

                // Insert values into respective columns
                for (size_t i = 0; i < columnNames.size(); i++)
                {
                    const hsql::Expr *expr = valueList->exprList->at(i);
                    const std::string &colName = columnNames[i];

                    // Get column data type
                    auto colType = table.getColumnType(colName);

                    // Insert based on data type
                    switch (colType)
                    {
                    case DataType::INT:
                    {
                        if (expr->type != hsql::kExprLiteralInt)
                        {
                            throw std::runtime_error("Type mismatch for column " + colName);
                        }
                        auto &col = static_cast<ColumnDataImpl<int> &>(table.getColumnData(colName));
                        col.append(expr->ival);
                        break;
                    }
                    case DataType::DOUBLE:
                    {
                        double value;
                        if (expr->type == hsql::kExprLiteralFloat)
                        {
                            value = expr->fval;
                        }
                        else if (expr->type == hsql::kExprLiteralInt)
                        {
                            value = static_cast<double>(expr->ival);
                        }
                        else
                        {
                            throw std::runtime_error("Type mismatch for column " + colName);
                        }
                        auto &col = static_cast<ColumnDataImpl<double> &>(table.getColumnData(colName));
                        col.append(value);
                        break;
                    }
                    case DataType::VARCHAR:
                    {
                        if (expr->type != hsql::kExprLiteralString)
                        {
                            throw std::runtime_error("Type mismatch for column " + colName);
                        }
                        auto &col = static_cast<ColumnDataImpl<std::string> &>(table.getColumnData(colName));
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
        }
        else
        {
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

    Table SQLQueryProcessor::executeSelectStatement(const hsql::SelectStatement *stmt, bool useGPU)
    {
        // Basic implementation to demonstrate integration
        Table resultTable;

        // Handle FROM clause (can be a table or a join)
        if (!stmt->fromTable)
        {
            throw std::runtime_error("FROM clause is required");
        }

        // Check if we're dealing with a table or a join
        if (stmt->fromTable->type == hsql::kTableName)
        {
            // Simple table reference
            resultTable = getTable(stmt->fromTable->name);
        }
        else if (stmt->fromTable->type == hsql::kTableJoin)
        {
            // Handle JOIN
            const hsql::JoinDefinition *join = stmt->fromTable->join;

            // Get the left and right tables
            Table leftTable = getTable(join->left->name);
            Table rightTable = getTable(join->right->name);

            // Translate the join condition
            auto joinCondition = translateWhereCondition(join->condition);

    

            // Execute the join
            // Join joinOp(leftTable, rightTable, *joinCondition, joinType);
            Join joinOp(leftTable, rightTable, *joinCondition);

            resultTable = joinOp.execute(useGPU && false);
        }
        // else if (stmt->fromTable->type == hsql::kTableCrossProduct)
        // {
        //     printf("Cross product detected\n");
        //     // Handle comma-separated tables (implicit join)
        //     // For each table in the cross product, join them one by one
        //     resultTable = getTable(stmt->fromTable->list->at(0)->name);

        //     // Start from the second table in the list
        //     for (size_t i = 1; i < stmt->fromTable->list->size(); i++)
        //     {
        //         Table rightTable = getTable(stmt->fromTable->list->at(i)->name);

        //         // Create a dummy condition that always evaluates to true
        //         // The real filtering will happen in the WHERE clause
        //         auto dummyCondition = ConditionBuilder::equals("1", "1");

        //         // Execute cross join (which is effectively what comma does)
        //         Join joinOp(resultTable, rightTable, *dummyCondition);
        //         resultTable = joinOp.execute(useGPU);
        //     }
        // }
        else
        {
            throw std::runtime_error("Unsupported FROM clause type");
        }

        // Apply WHERE condition if it exists
        if (stmt->whereClause)
        {
            auto condition = translateWhereCondition(stmt->whereClause);
            Select selectOp(resultTable, *condition);
            resultTable = selectOp.execute(useGPU);
        }

        // Check if this query uses aggregation
        bool hasAggregation = false;
        std::vector<Aggregation> aggregations;
        std::optional<std::string> groupByColumn = std::nullopt;

        // Check for GROUP BY clause
        if (stmt->groupBy)
        {
            hasAggregation = true;
            // Currently we only support a single GROUP BY column
            if (stmt->groupBy->columns->size() > 0)
            {
                const hsql::Expr *groupExpr = stmt->groupBy->columns->at(0);
                if (groupExpr->type == hsql::kExprColumnRef)
                {
                    groupByColumn = groupExpr->name;
                }
                else
                {
                    throw std::runtime_error("Only simple column references are supported in GROUP BY");
                }
            }
        }

        // Scan the select list for aggregation functions
        for (const hsql::Expr *expr : *stmt->selectList)
        {
            if (expr->type == hsql::kExprFunctionRef)
            {
                hasAggregation = true;

                // Get the function name and convert to upper case for case-insensitive comparison
                std::string funcName = expr->name;
                std::transform(funcName.begin(), funcName.end(), funcName.begin(), ::toupper);

                // Determine the aggregate function type
                AggregateFunction aggFunc;
                if (funcName == "COUNT")
                {
                    aggFunc = AggregateFunction::COUNT;
                }
                else if (funcName == "SUM")
                {
                    aggFunc = AggregateFunction::SUM;
                }
                else if (funcName == "AVG")
                {
                    aggFunc = AggregateFunction::AVG;
                }
                else if (funcName == "MIN")
                {
                    aggFunc = AggregateFunction::MIN;
                }
                else if (funcName == "MAX")
                {
                    aggFunc = AggregateFunction::MAX;
                }
                else
                {
                    throw std::runtime_error("Unsupported aggregate function: " + funcName);
                }

                // Get column name from function argument
                if (expr->exprList->size() != 1)
                {
                    throw std::runtime_error("Aggregate functions must have exactly one argument");
                }

                const hsql::Expr *argExpr = expr->exprList->at(0);
                std::string columnName;

                // Handle COUNT(*) specially
                if (aggFunc == AggregateFunction::COUNT && argExpr->type == hsql::kExprStar)
                {
                    // For COUNT(*), we can use any column as they all have the same count
                    columnName = resultTable.getColumnName(0);
                }
                else if (argExpr->type == hsql::kExprColumnRef)
                {
                    columnName = argExpr->name;
                }
                else
                {
                    throw std::runtime_error("Aggregate function argument must be a column reference");
                }

                // Get the alias if provided, otherwise use a default name
                std::string alias;
                if (expr->alias)
                {
                    alias = expr->alias;
                }
                else
                {
                    alias = funcName + "(" + columnName + ")";
                }

                aggregations.push_back(Aggregation(aggFunc, columnName, alias));
            }
            else if (expr->type == hsql::kExprColumnRef && hasAggregation)
            {
                // If we have aggregations, any column in SELECT must be in GROUP BY
                if (!groupByColumn.has_value() || expr->name != groupByColumn.value())
                {
                    throw std::runtime_error(
                        "Column " + std::string(expr->name) +
                        " must appear in GROUP BY clause or be used in an aggregate function");
                }
            }
        }

        // Apply aggregation if needed
        if (hasAggregation)
        {
            if (aggregations.empty())
            {
                throw std::runtime_error("GROUP BY used without aggregate functions");
            }

            Aggregator aggregator(resultTable, aggregations, groupByColumn);
            resultTable = aggregator.execute(useGPU && false);
        }
        // If no aggregation, handle normal SELECT (projection)
        else if (!stmt->selectList->empty() && (*stmt->selectList)[0]->type != hsql::kExprStar && false)
        {
            std::vector<std::string> projectColumns;
            for (auto &col : *stmt->selectList)
            {
                std::cout << "Column: " << col->name << std::endl;
            }
            for (const hsql::Expr *expr : *stmt->selectList)
            {
                if (expr->type == hsql::kExprColumnRef)
                {
                    std::string columnName;

                    // Handle table aliases in column references
                    if (expr->table)
                    {
                        // Just use the column name without the table alias
                        columnName = expr->name;
                    }
                    else
                    {
                        columnName = expr->name;
                    }

                    projectColumns.push_back(columnName);
                }
            }

            if (!projectColumns.empty())
            {
                resultTable.printTableInfo();
                Project projectOp(resultTable, projectColumns);
                resultTable = projectOp.execute(useGPU && false);
            }
        }

        // Apply ORDER BY if present
        if (stmt->order)
        {
            if (stmt->order->size() == 1)
            {
                // Single column ordering
                const hsql::OrderDescription *order = stmt->order->at(0);
                if (order->expr->type == hsql::kExprColumnRef)
                {
                    SortOrder sortOrder = order->type == hsql::kOrderAsc ? SortOrder::ASC : SortOrder::DESC;
                    OrderBy orderByOp(resultTable, order->expr->name, sortOrder);
                    resultTable = orderByOp.execute(useGPU && false);
                }
                else
                {
                    throw std::runtime_error("ORDER BY supports only column references");
                }
            }
            else if (stmt->order->size() > 1)
            {
                // Multi-column ordering
                std::vector<std::string> sortColumns;
                std::vector<SortOrder> sortOrders;

                for (const hsql::OrderDescription *order : *stmt->order)
                {
                    if (order->expr->type == hsql::kExprColumnRef)
                    {
                        sortColumns.push_back(order->expr->name);
                        sortOrders.push_back(order->type == hsql::kOrderAsc ? SortOrder::ASC : SortOrder::DESC);
                    }
                    else
                    {
                        throw std::runtime_error("ORDER BY supports only column references");
                    }
                }

                OrderBy orderByOp(resultTable, sortColumns, sortOrders);
                resultTable = orderByOp.execute(useGPU && false);
            }
        }

        return resultTable;
    }
    std::vector<std::string> SQLQueryProcessor::getTableNames() const
    {
        std::vector<std::string> tableNames;

        // Reserve space for efficiency
        tableNames.reserve(tables.size());

        // Extract all table names from the map
        for (const auto &[name, _] : tables)
        {
            tableNames.push_back(name);
        }

        return tableNames;
    }
    // You may also need to implement the translateWhereCondition method
    std::unique_ptr<Condition> SQLQueryProcessor::translateWhereCondition(const hsql::Expr *expr)
    {
        if (!expr)
            return nullptr;

        switch (expr->type)
        {
        case hsql::kExprOperator:
        {
            // Handle column-to-column comparison (for JOIN conditions)
            if (expr->expr->type == hsql::kExprColumnRef && expr->expr2->type == hsql::kExprColumnRef)
            {
                std::string leftColumn = expr->expr->name;
                std::string rightColumn = expr->expr2->name;

                std::cout << "Left column: " << leftColumn << std::endl;
                std::cout << "Right column: " << rightColumn << std::endl;

                switch (expr->opType)
                {
                case hsql::kOpEquals:
                    return ConditionBuilder::columnEquals(leftColumn, rightColumn);
                case hsql::kOpNotEquals:
                    return ConditionBuilder::columnNotEquals(leftColumn, rightColumn);
                case hsql::kOpLess:
                    return ConditionBuilder::columnLessThan(leftColumn, rightColumn);
                case hsql::kOpLessEq:
                    return ConditionBuilder::columnLessEqual(leftColumn, rightColumn);
                case hsql::kOpGreater:
                    return ConditionBuilder::columnGreaterThan(leftColumn, rightColumn);
                case hsql::kOpGreaterEq:
                    return ConditionBuilder::columnGreaterEqual(leftColumn, rightColumn);
                default:
                    throw std::runtime_error("Unsupported column comparison operator in JOIN condition");
                }
            }
            if (expr->expr2->type == hsql::kExprLiteralString)
            {
                // Get the column name from expr
                std::string columnName;
                if (expr->expr->type == hsql::kExprColumnRef)
                {
                    // Handle table aliases if present
                    if (expr->expr->table)
                    {
                        columnName = expr->expr->name;
                    }
                    else
                    {
                        columnName = expr->expr->name;
                    }
                }
                else
                {
                    throw std::runtime_error("Left side of condition must be a column reference");
                }

                std::string value = expr->expr2->name;

                // If this looks like a DateTime value, strip quotes
                if (value.length() >= 2 && (value[0] == '\'' || value[0] == '"') &&
                    (value[value.length() - 1] == '\'' || value[value.length() - 1] == '"'))
                {
                    value = value.substr(1, value.length() - 2);
                }

                // Create the condition
                switch (expr->opType)
                {
                case hsql::kOpEquals:
                    return ConditionBuilder::equals(columnName, value);
                case hsql::kOpNotEquals:
                    return ConditionBuilder::notEquals(columnName, value);
                case hsql::kOpLess:
                    return ConditionBuilder::lessThan(columnName, value);
                case hsql::kOpLessEq:
                    return ConditionBuilder::lessEqual(columnName, value);
                case hsql::kOpGreater:
                    return ConditionBuilder::greaterThan(columnName, value);
                case hsql::kOpGreaterEq:
                    return ConditionBuilder::greaterEqual(columnName, value);
                case hsql::kOpLike:
                    return ConditionBuilder::like(columnName, value);
                default:
                    throw std::runtime_error("Unsupported string operator in WHERE clause");
                }
            }
            // Handle binary operators
            if (expr->expr && expr->expr2)
            {
                // Get the column name (left side of condition)
                std::string columnName;
                if (expr->expr->type == hsql::kExprColumnRef)
                {
                    // Handle table aliases if present (e.g., e.id)
                    std::cout << "Column reference: " << expr->expr->name << std::endl;
                    if (expr->expr->table)
                    {

                        columnName = expr->expr->name;
                    }
                    else
                    {
                        columnName = expr->expr->name;
                    }
                }
                else
                {
                    throw std::runtime_error("Left side of condition must be a column reference");
                }

                // Handle based on operator type
                switch (expr->opType)
                {
                case hsql::kOpEquals:
                {
                    // Handle different types of right operands
                    if (expr->expr2->type == hsql::kExprLiteralString)
                    {
                        return ConditionBuilder::equals(columnName, expr->expr2->name);
                    }
                    else if (expr->expr2->type == hsql::kExprLiteralInt)
                    {
                        return ConditionBuilder::equals(columnName, std::to_string(expr->expr2->ival));
                    }
                    else if (expr->expr2->type == hsql::kExprLiteralFloat)
                    {
                        return ConditionBuilder::equals(columnName, std::to_string(expr->expr2->fval));
                    }
                    break;
                }
                case hsql::kOpGreater:
                {
                    if (expr->expr2->type == hsql::kExprLiteralInt)
                    {
                        return ConditionBuilder::greaterThan(columnName, std::to_string(expr->expr2->ival));
                    }
                    else if (expr->expr2->type == hsql::kExprLiteralFloat)
                    {
                        return ConditionBuilder::greaterThan(columnName, std::to_string(expr->expr2->fval));
                    }
                    break;
                }
                case hsql::kOpLess:
                {
                    if (expr->expr2->type == hsql::kExprLiteralInt)
                    {
                        return ConditionBuilder::lessThan(columnName, std::to_string(expr->expr2->ival));
                    }
                    else if (expr->expr2->type == hsql::kExprLiteralFloat)
                    {
                        return ConditionBuilder::lessThan(columnName, std::to_string(expr->expr2->fval));
                    }
                    break;
                }
                case hsql::kOpAnd:
                {
                    auto leftCond = translateWhereCondition(expr->expr);
                    auto rightCond = translateWhereCondition(expr->expr2);
                    return ConditionBuilder::And(std::move(leftCond), std::move(rightCond));
                }
                case hsql::kOpOr:
                {
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