#include "Operations/Project.hpp"
#include "Utilities/Logger.hpp"
#include "Utilities/ErrorHandling.hpp"
#include <algorithm>

namespace SQLQueryProcessor {

std::shared_ptr<Table> Project::executeCPU(
    const std::shared_ptr<Table>& inputTable,
    const std::vector<std::string>& columnNames,
    const std::vector<int>& columnIndices) {
    
    Logger::debug("Executing CPU projection operation");
    
    if (!inputTable) {
        throw ExecutionException("Project operation received null input table");
    }
    
    // Prepare column indices if not provided
    std::vector<int> actualColumnIndices = columnIndices;
    if (actualColumnIndices.empty() && !columnNames.empty()) {
        actualColumnIndices = prepareColumnIndices(inputTable, columnNames);
    }
    
    // If no columns specified, select all columns
    if (actualColumnIndices.empty()) {
        for (size_t i = 0; i < inputTable->getColumnCount(); ++i) {
            actualColumnIndices.push_back(i);
        }
    }
    
    // Create output table
    auto resultTable = std::make_shared<Table>("project_result");
    
    // Add columns to result table
    for (int colIndex : actualColumnIndices) {
        if (colIndex < 0 || colIndex >= static_cast<int>(inputTable->getColumnCount())) {
            throw ExecutionException("Invalid column index: " + std::to_string(colIndex));
        }
        
        const std::string& columnName = inputTable->getColumnNames()[colIndex];
        bool isPrimary = inputTable->isColumnPrimaryKey(colIndex);
        bool isForeign = inputTable->isColumnForeignKey(colIndex);
        resultTable->addColumn(columnName, isPrimary, isForeign);
    }
    
    // Copy data rows
    for (size_t rowIndex = 0; rowIndex < inputTable->getRowCount(); ++rowIndex) {
        std::vector<std::string> newRow;
        newRow.reserve(actualColumnIndices.size());
        
        for (int colIndex : actualColumnIndices) {
            newRow.push_back(inputTable->getValue(rowIndex, colIndex));
        }
        
        resultTable->addRow(newRow);
    }
    
    Logger::debug("CPU projection completed, result has " + 
                 std::to_string(resultTable->getRowCount()) + " rows and " + 
                 std::to_string(resultTable->getColumnCount()) + " columns");
    
    return resultTable;
}

std::shared_ptr<Table> Project::executeGPU(
    const std::shared_ptr<Table>& inputTable,
    const std::vector<std::string>& columnNames,
    const std::vector<int>& columnIndices) {
    
    Logger::debug("Executing GPU projection operation");
    
    if (!inputTable) {
        throw ExecutionException("Project operation received null input table");
    }
    
    // Prepare column indices if not provided
    std::vector<int> actualColumnIndices = columnIndices;
    if (actualColumnIndices.empty() && !columnNames.empty()) {
        actualColumnIndices = prepareColumnIndices(inputTable, columnNames);
    }
    
    // If no columns specified, select all columns
    if (actualColumnIndices.empty()) {
        for (size_t i = 0; i < inputTable->getColumnCount(); ++i) {
            actualColumnIndices.push_back(i);
        }
    }
    
    // If table is small, use CPU projection
    if (inputTable->getRowCount() < 1000) {
        Logger::debug("Small table detected, using CPU projection instead");
        return executeCPU(inputTable, columnNames, actualColumnIndices);
    }
    
    // For now, return CPU implementation
    // In a real implementation, we would use CUDA for the projection
    Logger::warning("GPU projection not fully implemented, falling back to CPU");
    
    return executeCPU(inputTable, columnNames, actualColumnIndices);
}

std::vector<int> Project::prepareColumnIndices(
    const std::shared_ptr<Table>& inputTable,
    const std::vector<std::string>& columnNames) {
    
    std::vector<int> indices;
    indices.reserve(columnNames.size());
    
    for (const auto& columnName : columnNames) {
        // Handle table-qualified column names
        size_t dotPos = columnName.find('.');
        if (dotPos != std::string::npos) {
            std::string tableName = columnName.substr(0, dotPos);
            std::string colName = columnName.substr(dotPos + 1);
            
            // Check if tableName matches inputTable's name
            if (tableName == inputTable->getName()) {
                int index = inputTable->getColumnIndex(colName);
                if (index < 0) {
                    throw ExecutionException("Column not found: " + colName + " in table " + tableName);
                }
                indices.push_back(index);
            } else {
                // Look for the column as "tableName.columnName" in the input table
                int index = inputTable->getColumnIndex(columnName);
                if (index < 0) {
                    throw ExecutionException("Column not found: " + columnName);
                }
                indices.push_back(index);
            }
        } else {
            // Regular column name
            int index = inputTable->getColumnIndex(columnName);
            if (index < 0) {
                throw ExecutionException("Column not found: " + columnName);
            }
            indices.push_back(index);
        }
    }
    
    return indices;
}

} // namespace SQLQueryProcessor