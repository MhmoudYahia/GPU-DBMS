#include "Operations/Filter.hpp"
#include "Utilities/Logger.hpp"
#include "Utilities/ErrorHandling.hpp"
#include "Utilities/StringUtils.hpp"
#include <algorithm>
#include <cuda_runtime.h>

namespace SQLQueryProcessor {

void Filter::addCondition(const FilterCondition& condition, LogicalOperator logicOp) {
    if (!conditions.empty()) {
        logicalOperators.push_back(logicOp);
    }
    conditions.push_back(condition);
}

void Filter::clearConditions() {
    conditions.clear();
    logicalOperators.clear();
}

std::shared_ptr<Table> Filter::executeCPU(const std::shared_ptr<Table>& inputTable) {
    Logger::debug("Executing CPU filter operation with " + 
                 std::to_string(conditions.size()) + " condition(s)");
    
    if (!inputTable) {
        throw ExecutionException("Filter operation received null input table");
    }
    
    if (conditions.empty()) {
        // No conditions, return a copy of the input table
        auto resultTable = inputTable->createSimilarTable("filter_result");
        for (size_t i = 0; i < inputTable->getRowCount(); ++i) {
            const auto& row = inputTable->getData()[i];
            resultTable->addRow(row);
        }
        return resultTable;
    }
    
    // Create output table with same schema
    auto resultTable = inputTable->createSimilarTable("filter_result");
    
    // Apply filter to each row
    for (size_t i = 0; i < inputTable->getRowCount(); ++i) {
        if (evaluateRowCPU(inputTable->getData()[i], inputTable)) {
            resultTable->addRow(inputTable->getData()[i]);
        }
    }
    
    Logger::debug("CPU filter operation completed, result has " + 
                 std::to_string(resultTable->getRowCount()) + " rows");
    
    return resultTable;
}

std::shared_ptr<Table> Filter::executeGPU(const std::shared_ptr<Table>& inputTable) {
    Logger::debug("Executing GPU filter operation with " + 
                 std::to_string(conditions.size()) + " condition(s)");
    
    if (!inputTable) {
        throw ExecutionException("Filter operation received null input table");
    }
    
    if (conditions.empty()) {
        // No conditions, return a copy of the input table
        auto resultTable = inputTable->createSimilarTable("filter_result");
        for (size_t i = 0; i < inputTable->getRowCount(); ++i) {
            const auto& row = inputTable->getData()[i];
            resultTable->addRow(row);
        }
        return resultTable;
    }
    
    // For small tables, CPU might be faster
    if (inputTable->getRowCount() < 10000) {
        Logger::debug("Table too small for GPU filter, falling back to CPU");
        return executeCPU(inputTable);
    }
    
    // Create output table with same schema
    auto resultTable = inputTable->createSimilarTable("filter_result");
    
    // Prepare column indices for conditions
    std::vector<int> columnIndices;
    for (const auto& condition : conditions) {
        int colIndex = inputTable->getColumnIndex(condition.columnName);
        if (colIndex < 0) {
            throw ExecutionException("Column not found in filter: " + condition.columnName);
        }
        columnIndices.push_back(colIndex);
    }
    
    // In a real implementation, we would allocate GPU memory and execute the kernel
    
    // For now, fall back to CPU implementation as a placeholder
    // This would be replaced with actual CUDA implementation in the real code
    for (size_t i = 0; i < inputTable->getRowCount(); ++i) {
        if (evaluateRowCPU(inputTable->getData()[i], inputTable)) {
            resultTable->addRow(inputTable->getData()[i]);
        }
    }
    
    Logger::debug("GPU filter operation completed (CPU fallback), result has " + 
                 std::to_string(resultTable->getRowCount()) + " rows");
    
    return resultTable;
}

bool Filter::evaluateConditionCPU(const std::string& value, const FilterCondition& condition) {
    // Try to parse as numeric first for numeric comparisons
    bool leftIsNumeric = false;
    bool rightIsNumeric = false;
    double leftNumeric = 0.0;
    double rightNumeric = 0.0;
    
    leftIsNumeric = StringUtils::tryParseDouble(value, leftNumeric);
    rightIsNumeric = StringUtils::tryParseDouble(condition.value, rightNumeric);
    
    // If both are numeric, do numeric comparison
    if (leftIsNumeric && rightIsNumeric) {
        switch (condition.compOp) {
            case ComparisonOperator::EQUAL:
                return leftNumeric == rightNumeric;
            case ComparisonOperator::NOT_EQUAL:
                return leftNumeric != rightNumeric;
            case ComparisonOperator::GREATER:
                return leftNumeric > rightNumeric;
            case ComparisonOperator::GREATER_EQUAL:
                return leftNumeric >= rightNumeric;
            case ComparisonOperator::LESS:
                return leftNumeric < rightNumeric;
            case ComparisonOperator::LESS_EQUAL:
                return leftNumeric <= rightNumeric;
            default:
                return false;
        }
    } else {
        // String comparison
        int cmpResult = value.compare(condition.value);
        
        switch (condition.compOp) {
            case ComparisonOperator::EQUAL:
                return cmpResult == 0;
            case ComparisonOperator::NOT_EQUAL:
                return cmpResult != 0;
            case ComparisonOperator::GREATER:
                return cmpResult > 0;
            case ComparisonOperator::GREATER_EQUAL:
                return cmpResult >= 0;
            case ComparisonOperator::LESS:
                return cmpResult < 0;
            case ComparisonOperator::LESS_EQUAL:
                return cmpResult <= 0;
            default:
                return false;
        }
    }
}

bool Filter::evaluateRowCPU(const std::vector<std::string>& row, const std::shared_ptr<Table>& table) {
    if (conditions.empty()) {
        return true;
    }
    
    // Evaluate first condition
    bool result = false;
    int columnIndex = table->getColumnIndex(conditions[0].columnName);
    
    if (columnIndex >= 0 && columnIndex < static_cast<int>(row.size())) {
        result = evaluateConditionCPU(row[columnIndex], conditions[0]);
    }
    
    // Evaluate remaining conditions
    for (size_t i = 1; i < conditions.size(); ++i) {
        columnIndex = table->getColumnIndex(conditions[i].columnName);
        
        if (columnIndex < 0 || columnIndex >= static_cast<int>(row.size())) {
            continue;
        }
        
        bool condResult = evaluateConditionCPU(row[columnIndex], conditions[i]);
        
        // Apply logical operator
        if (logicalOperators[i - 1] == LogicalOperator::AND) {
            result = result && condResult;
        } else {  // OR
            result = result || condResult;
        }
    }
    
    return result;
}

// CUDA kernel implementations would go in a .cu file
/*
__device__ bool evaluateConditionGPU(const char* value, const char* conditionValue, int compOp) {
    // Try to parse as numeric
    bool leftIsNumeric = true;
    bool rightIsNumeric = true;
    double leftNumeric = 0.0;
    double rightNumeric = 0.0;
    
    // Converting string to numeric on GPU
    char* endPtr;
    leftNumeric = strtod(value, &endPtr);
    if (endPtr == value) leftIsNumeric = false;
    
    rightNumeric = strtod(conditionValue, &endPtr);
    if (endPtr == conditionValue) rightIsNumeric = false;
    
    if (leftIsNumeric && rightIsNumeric) {
        // Numeric comparison
        switch (compOp) {
            case 0: return leftNumeric == rightNumeric;  // EQUAL
            case 1: return leftNumeric != rightNumeric;  // NOT_EQUAL
            case 2: return leftNumeric > rightNumeric;   // GREATER
            case 3: return leftNumeric >= rightNumeric;  // GREATER_EQUAL
            case 4: return leftNumeric < rightNumeric;   // LESS
            case 5: return leftNumeric <= rightNumeric;  // LESS_EQUAL
            default: return false;
        }
    } else {
        // String comparison
        int cmpResult = strcmp(value, conditionValue);
        
        switch (compOp) {
            case 0: return cmpResult == 0;  // EQUAL
            case 1: return cmpResult != 0;  // NOT_EQUAL
            case 2: return cmpResult > 0;   // GREATER
            case 3: return cmpResult >= 0;  // GREATER_EQUAL
            case 4: return cmpResult < 0;   // LESS
            case 5: return cmpResult <= 0;  // LESS_EQUAL
            default: return false;
        }
    }
}

__global__ void filterKernel(const char** inputData, char** outputData, bool* resultFlags,
                           int* columnIndices, int* compOps, const char** compareValues,
                           int* logicOps, int numConditions, int numRows, int numColumns) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRows) {
        // Evaluate first condition
        bool result = false;
        if (numConditions > 0) {
            int columnIndex = columnIndices[0];
            const char* cellValue = inputData[idx * numColumns + columnIndex];
            result = evaluateConditionGPU(cellValue, compareValues[0], compOps[0]);
            
            // Evaluate remaining conditions
            for (int i = 1; i < numConditions; i++) {
                columnIndex = columnIndices[i];
                cellValue = inputData[idx * numColumns + columnIndex];
                bool condResult = evaluateConditionGPU(cellValue, compareValues[i], compOps[i]);
                
                // Apply logical operator
                if (logicOps[i-1] == 0) {  // AND
                    result = result && condResult;
                } else {  // OR
                    result = result || condResult;
                }
            }
        }
        
        // Store the result
        resultFlags[idx] = result;
        
        // In a stream compaction approach, we would not copy data here
        // Instead, we'd use a separate kernel after a prefix sum to copy only matching rows
    }
}
*/

} // namespace SQLQueryProcessor