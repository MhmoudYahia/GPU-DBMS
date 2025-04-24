#include "QueryProcessing/PlanBuilder.hpp"
#include "Utilities/Logger.hpp"
#include "Utilities/ErrorHandling.hpp"
#include <algorithm>
#include <unordered_set>

namespace SQLQueryProcessor {

PlanBuilder::PlanBuilder(StorageManager& storageManager, ExecutionStrategy strategy)
    : storageManager(storageManager), executionStrategy(strategy) {
    
    Logger::debug("PlanBuilder initialized with strategy: " + 
                 (strategy == ExecutionStrategy::SEQUENTIAL ? "SEQUENTIAL" : 
                 (strategy == ExecutionStrategy::PIPELINED ? "PIPELINED" : "PARALLEL")));
}

std::shared_ptr<Table> PlanBuilder::buildAndExecutePlan(
    const ASTProcessor::QueryInfo& queryInfo, 
    bool useGPU,
    bool useStreaming) {
    
    Logger::debug("Building execution plan" + std::string(useGPU ? " (GPU)" : " (CPU)"));
    
    // Step 1: Get base tables
    std::vector<std::shared_ptr<Table>> baseTables;
    for (const auto& tableName : queryInfo.fromTables) {
        std::shared_ptr<Table> table = executeTableScan(tableName);
        if (!table) {
            throw ExecutionException("Table not found: " + tableName);
        }
        baseTables.push_back(table);
    }
    
    // Step 2: Execute joins to create a single joined table
    std::shared_ptr<Table> joinedTable;
    if (baseTables.size() == 1) {
        joinedTable = baseTables[0];
    } else if (!queryInfo.joins.empty()) {
        joinedTable = executeJoins(baseTables, queryInfo.joins, useGPU);
    } else {
        throw ExecutionException("Multiple tables specified without join conditions");
    }
    
    // Step 3: Execute filters
    std::shared_ptr<Table> filteredTable;
    if (!queryInfo.whereConditions.empty()) {
        filteredTable = executeFilter(joinedTable, queryInfo.whereConditions, queryInfo.whereLogicalOps, useGPU);
    } else {
        filteredTable = joinedTable;
    }
    
    // Step 4: Execute group by and aggregation if present
    std::shared_ptr<Table> aggregatedTable;
    if (!queryInfo.groupByColumns.empty() || 
        std::any_of(queryInfo.selectColumns.begin(), queryInfo.selectColumns.end(),
                   [](const ASTProcessor::ColumnInfo& col) { return col.isAggregation; })) {
        
        aggregatedTable = executeGroupByAndAggregation(filteredTable, queryInfo.groupByColumns, queryInfo.selectColumns, useGPU);
    } else {
        aggregatedTable = filteredTable;
    }
    
    // Step 5: Execute order by if present
    std::shared_ptr<Table> orderedTable;
    if (!queryInfo.orderByColumns.empty()) {
        orderedTable = executeOrderBy(aggregatedTable, queryInfo.orderByColumns, useGPU);
    } else {
        orderedTable = aggregatedTable;
    }
    
    // Step 6: Execute projection to get final columns
    std::shared_ptr<Table> resultTable = executeProjection(orderedTable, queryInfo.selectColumns, useGPU);
    
    Logger::debug("Execution plan completed, result has " + 
                 std::to_string(resultTable->getRowCount()) + " rows and " +
                 std::to_string(resultTable->getColumnCount()) + " columns");
    
    return resultTable;
}

void PlanBuilder::setExecutionStrategy(ExecutionStrategy strategy) {
    executionStrategy = strategy;
    Logger::debug("PlanBuilder strategy set to: " + 
                 (strategy == ExecutionStrategy::SEQUENTIAL ? "SEQUENTIAL" : 
                 (strategy == ExecutionStrategy::PIPELINED ? "PIPELINED" : "PARALLEL")));
}

PlanBuilder::ExecutionStrategy PlanBuilder::getExecutionStrategy() const {
    return executionStrategy;
}

std::shared_ptr<Table> PlanBuilder::executeTableScan(const std::string& tableName) {
    if (tableName == "SUBQUERY_RESULT") {
        // Special handling for subquery results
        return storageManager.getTemporaryTable("temp_subquery");
    }
    
    return storageManager.getTable(tableName);
}

std::shared_ptr<Table> PlanBuilder::executeJoins(
    const std::vector<std::shared_ptr<Table>>& tables,
    const std::vector<ASTProcessor::JoinInfo>& joinInfos,
    bool useGPU) {
    
    if (tables.size() < 2 || joinInfos.empty()) {
        throw ExecutionException("Invalid join operation");
    }
    
    // Build a table map for quick lookup
    std::unordered_map<std::string, std::shared_ptr<Table>> tableMap;
    for (const auto& table : tables) {
        tableMap[table->getName()] = table;
    }
    
    // Start with the first join
    const auto& firstJoin = joinInfos[0];
    auto leftTable = tableMap[firstJoin.leftTable];
    auto rightTable = tableMap[firstJoin.rightTable];
    
    if (!leftTable || !rightTable) {
        throw ExecutionException("Join table not found");
    }
    
    // Execute the first join
    std::shared_ptr<Table> result;
    if (useGPU) {
        result = joinOp.executeGPU(leftTable, rightTable, firstJoin.leftColumn, firstJoin.rightColumn);
    } else {
        result = joinOp.executeCPU(leftTable, rightTable, firstJoin.leftColumn, firstJoin.rightColumn);
    }
    
    // Add the result to the table map for subsequent joins
    tableMap["joined_result"] = result;
    
    // Execute remaining joins
    for (size_t i = 1; i < joinInfos.size(); ++i) {
        const auto& join = joinInfos[i];
        
        // Find the tables to join
        std::string leftTableName = join.leftTable;
        std::string rightTableName = join.rightTable;
        
        // If one of the tables is from a previous join, use the joined result
        if (!tableMap.count(leftTableName) || !tableMap.count(rightTableName)) {
            leftTable = tableMap["joined_result"];
            rightTable = tableMap.count(leftTableName) ? tableMap[rightTableName] : tableMap[leftTableName];
            
            // Adjust column names to match the joined table
            std::string leftColumn = join.leftColumn;
            std::string rightColumn = join.rightColumn;
            
            if (!tableMap.count(leftTableName)) {
                leftColumn = join.leftColumn;
            } else {
                rightColumn = join.rightColumn;
            }
            
            if (useGPU) {
                result = joinOp.executeGPU(leftTable, rightTable, leftColumn, rightColumn);
            } else {
                result = joinOp.executeCPU(leftTable, rightTable, leftColumn, rightColumn);
            }
        } else {
            leftTable = tableMap[leftTableName];
            rightTable = tableMap[rightTableName];
            
            if (useGPU) {
                result = joinOp.executeGPU(leftTable, rightTable, join.leftColumn, join.rightColumn);
            } else {
                result = joinOp.executeCPU(leftTable, rightTable, join.leftColumn, join.rightColumn);
            }
        }
        
        // Update the joined result for the next iteration
        tableMap["joined_result"] = result;
    }
    
    return result;
}

std::shared_ptr<Table> PlanBuilder::executeFilter(
    const std::shared_ptr<Table>& table,
    const std::vector<FilterCondition>& conditions,
    const std::vector<LogicalOperator>& logicalOps,
    bool useGPU) {
    
    if (!table) {
        throw ExecutionException("Filter operation received null input table");
    }
    
    if (conditions.empty()) {
        return table;
    }
    
    // Add conditions to the filter operator
    filterOp.clearConditions();
    
    for (size_t i = 0; i < conditions.size(); ++i) {
        LogicalOperator logicOp = LogicalOperator::AND;
        if (i > 0 && i - 1 < logicalOps.size()) {
            logicOp = logicalOps[i - 1];
        }
        
        filterOp.addCondition(conditions[i], logicOp);
    }
    
    // Execute filter
    if (useGPU) {
        return filterOp.executeGPU(table);
    } else {
        return filterOp.executeCPU(table);
    }
}

std::shared_ptr<Table> PlanBuilder::executeGroupByAndAggregation(
    const std::shared_ptr<Table>& table,
    const std::vector<std::string>& groupByColumns,
    const std::vector<ASTProcessor::ColumnInfo>& selectColumns,
    bool useGPU) {
    
    if (!table) {
        throw ExecutionException("GroupBy/Aggregation operation received null input table");
    }
    
    // Set up aggregator
    aggregatorOp.clearAggregations();
    aggregatorOp.clearGroupByColumns();
    
    // Add group by columns
    for (const auto& column : groupByColumns) {
        aggregatorOp.addGroupByColumn(column);
    }
    
    // Add aggregations from select columns
    for (const auto& colInfo : selectColumns) {
        if (colInfo.isAggregation) {
            aggregatorOp.addAggregation(colInfo.aggrFunc, colInfo.column, colInfo.alias);
        }
    }
    
    // Execute aggregation
    if (useGPU) {
        return aggregatorOp.executeGPU(table);
    } else {
        return aggregatorOp.executeCPU(table);
    }
}

std::shared_ptr<Table> PlanBuilder::executeOrderBy(
    const std::shared_ptr<Table>& table,
    const std::vector<std::pair<std::string, SortOrder>>& orderByColumns,
    bool useGPU) {
    
    if (!table) {
        throw ExecutionException("OrderBy operation received null input table");
    }
    
    if (orderByColumns.empty()) {
        return table;
    }
    
    // Set up order by
    orderByOp.clearSortKeys();
    
    // Add sort keys
    for (const auto& [column, order] : orderByColumns) {
        orderByOp.addSortKey(column, order);
    }
    
    // Execute order by
    if (useGPU) {
        return orderByOp.executeGPU(table);
    } else {
        return orderByOp.executeCPU(table);
    }
}

std::shared_ptr<Table> PlanBuilder::executeProjection(
    const std::shared_ptr<Table>& table,
    const std::vector<ASTProcessor::ColumnInfo>& selectColumns,
    bool useGPU) {
    
    if (!table) {
        throw ExecutionException("Projection operation received null input table");
    }
    
    // Special case: SELECT *
    if (selectColumns.size() == 1 && selectColumns[0].column == "*") {
        return table;
    }
    
    // Build list of columns to project
    std::vector<std::string> columnNames;
    for (const auto& colInfo : selectColumns) {
        // Skip aggregations as they have been handled in the aggregation step
        if (!colInfo.isAggregation) {
            // Handle table-qualified columns
            if (!colInfo.table.empty()) {
                columnNames.push_back(colInfo.table + "." + colInfo.column);
            } else {
                columnNames.push_back(colInfo.column);
            }
        }
    }
    
    // If we're only selecting aggregations, the projection is already done
    if (columnNames.empty() && 
        std::all_of(selectColumns.begin(), selectColumns.end(),
                   [](const ASTProcessor::ColumnInfo& col) { return col.isAggregation; })) {
        return table;
    }
    
    // Execute projection
    std::shared_ptr<Table> result;
    if (useGPU) {
        result = projectOp.executeGPU(table, columnNames);
    } else {
        result = projectOp.executeCPU(table, columnNames);
    }
    
    // Rename columns according to aliases
    if (result) {
        auto resultWithAliases = std::make_shared<Table>("projection_result");
        
        // Add columns with aliases
        for (size_t i = 0; i < selectColumns.size(); ++i) {
            if (i < result->getColumnCount()) {
                resultWithAliases->addColumn(selectColumns[i].alias, false, false);
            }
        }
        
        // Copy data rows
        for (size_t i = 0; i < result->getRowCount(); ++i) {
            std::vector<std::string> row;
            for (size_t j = 0; j < result->getColumnCount(); ++j) {
                row.push_back(result->getValue(i, j));
            }
            resultWithAliases->addRow(row);
        }
        
        return resultWithAliases;
    }
    
    return result;
}

} // namespace SQLQueryProcessor