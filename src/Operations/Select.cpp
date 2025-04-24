#include "Operations/Select.hpp"
#include "Utilities/Logger.hpp"
#include "Utilities/ErrorHandling.hpp"
#include <algorithm>

namespace SQLQueryProcessor {

std::shared_ptr<Table> Select::executeCPU(
    const std::shared_ptr<Table>& inputTable,
    const std::vector<std::string>& columnNames,
    const std::vector<int>& columnIndices) {
    
    Logger::debug("Executing CPU select operation");
    
    if (!inputTable) {
        throw ExecutionException("Select operation received null input table");
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
    
    // Validate column indices
    for (int index : actualColumnIndices) {
        if (index < 0 || index >= static_cast<int>(inputTable->getColumnCount())) {
            throw ExecutionException("Invalid column index in select operation: " + std::to_string(index));
        }
    }
    
    // Create output table
    auto resultTable = std::make_shared<Table>("select_result");
    
    // Add columns to result table
    for (int colIndex : actualColumnIndices) {
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
    
    Logger::debug("CPU select operation completed, result table has " + 
                 std::to_string(resultTable->getRowCount()) + " rows and " + 
                 std::to_string(resultTable->getColumnCount()) + " columns");
    
    return resultTable;
}

std::shared_ptr<Table> Select::executeGPU(
    const std::shared_ptr<Table>& inputTable,
    const std::vector<std::string>& columnNames,
    const std::vector<int>& columnIndices) {
    
    Logger::debug("Executing GPU select operation");
    
    if (!inputTable) {
        throw ExecutionException("Select operation received null input table");
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
    
    // Get dimensions
    size_t numRows = inputTable->getRowCount();
    size_t numCols = inputTable->getColumnCount();
    size_t numOutputCols = actualColumnIndices.size();
    
    if (numRows == 0 || numOutputCols == 0) {
        // Special case for empty input
        auto resultTable = std::make_shared<Table>("select_result");
        for (int colIndex : actualColumnIndices) {
            const std::string& columnName = inputTable->getColumnNames()[colIndex];
            bool isPrimary = inputTable->isColumnPrimaryKey(colIndex);
            bool isForeign = inputTable->isColumnForeignKey(colIndex);
            resultTable->addColumn(columnName, isPrimary, isForeign);
        }
        return resultTable;
    }
    
    // Create arrays for input and output data
    const auto& inputData = inputTable->getData();
    
    // Create result table
    auto resultTable = std::make_shared<Table>("select_result");
    
    // Add columns to result table
    for (int colIndex : actualColumnIndices) {
        const std::string& columnName = inputTable->getColumnNames()[colIndex];
        bool isPrimary = inputTable->isColumnPrimaryKey(colIndex);
        bool isForeign = inputTable->isColumnForeignKey(colIndex);
        resultTable->addColumn(columnName, isPrimary, isForeign);
    }
    
    // Prepare GPU memory for input data
    char** d_inputData;
    char** d_outputData;
    int* d_columnIndices;
    
    // Allocate device memory
    cudaMalloc(&d_inputData, sizeof(char*) * numRows * numCols);
    cudaMalloc(&d_outputData, sizeof(char*) * numRows * numOutputCols);
    cudaMalloc(&d_columnIndices, sizeof(int) * numOutputCols);
    
    // Copy data to device
    // In a real implementation, we would also copy the actual string data
    // For now, we'll proceed with the CPU implementation
    
    cudaFree(d_inputData);
    cudaFree(d_outputData);
    cudaFree(d_columnIndices);
    
    // CPU fallback for now
    return executeCPU(inputTable, columnNames, actualColumnIndices);
}

std::vector<int> Select::prepareColumnIndices(
    const std::shared_ptr<Table>& inputTable,
    const std::vector<std::string>& columnNames) {
    
    std::vector<int> indices;
    indices.reserve(columnNames.size());
    
    for (const auto& columnName : columnNames) {
        int index = inputTable->getColumnIndex(columnName);
        if (index < 0) {
            throw ExecutionException("Column not found: " + columnName);
        }
        indices.push_back(index);
    }
    
    return indices;
}

// CUDA kernel implementation would go here in a .cu file

} // namespace SQLQueryProcessor