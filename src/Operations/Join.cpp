#include "Operations/Join.hpp"
#include "Utilities/Logger.hpp"
#include "Utilities/ErrorHandling.hpp"
#include "Utilities/StringUtils.hpp"
#include <unordered_map>
#include <algorithm>
#include <cuda_runtime.h>

namespace SQLQueryProcessor {

std::shared_ptr<Table> Join::executeCPU(
    const std::shared_ptr<Table>& leftTable,
    const std::shared_ptr<Table>& rightTable,
    const std::string& leftColumn,
    const std::string& rightColumn,
    JoinType joinType) {
    
    Logger::debug("Executing CPU join operation");
    
    if (!leftTable || !rightTable) {
        throw ExecutionException("Join operation received null input table");
    }
    
    // Get column indices
    int leftColumnIndex = leftTable->getColumnIndex(leftColumn);
    int rightColumnIndex = rightTable->getColumnIndex(rightColumn);
    
    if (leftColumnIndex < 0) {
        throw ExecutionException("Left join column not found: " + leftColumn);
    }
    
    if (rightColumnIndex < 0) {
        throw ExecutionException("Right join column not found: " + rightColumn);
    }
    
    // Select the appropriate join implementation based on join type
    switch (joinType) {
        case JoinType::INNER:
            return innerJoinCPU(leftTable, rightTable, leftColumnIndex, rightColumnIndex);
        case JoinType::LEFT:
        case JoinType::RIGHT:
        case JoinType::FULL:
            // These would be implemented similarly but with different NULL handling
            throw ExecutionException("This join type is not yet implemented");
        default:
            throw ExecutionException("Unknown join type");
    }
}

std::shared_ptr<Table> Join::executeGPU(
    const std::shared_ptr<Table>& leftTable,
    const std::shared_ptr<Table>& rightTable,
    const std::string& leftColumn,
    const std::string& rightColumn,
    JoinType joinType) {
    
    Logger::debug("Executing GPU join operation");
    
    if (!leftTable || !rightTable) {
        throw ExecutionException("Join operation received null input table");
    }
    
    // Get column indices
    int leftColumnIndex = leftTable->getColumnIndex(leftColumn);
    int rightColumnIndex = rightTable->getColumnIndex(rightColumn);
    
    if (leftColumnIndex < 0) {
        throw ExecutionException("Left join column not found: " + leftColumn);
    }
    
    if (rightColumnIndex < 0) {
        throw ExecutionException("Right join column not found: " + rightColumn);
    }
    
    // Select the appropriate join implementation based on join type
    switch (joinType) {
        case JoinType::INNER:
            return innerJoinGPU(leftTable, rightTable, leftColumnIndex, rightColumnIndex);
        case JoinType::LEFT:
        case JoinType::RIGHT:
        case JoinType::FULL:
            // These would be implemented similarly but with different NULL handling
            throw ExecutionException("This join type is not yet implemented on GPU");
        default:
            throw ExecutionException("Unknown join type");
    }
}

std::shared_ptr<Table> Join::innerJoinCPU(
    const std::shared_ptr<Table>& leftTable,
    const std::shared_ptr<Table>& rightTable,
    int leftColumnIndex,
    int rightColumnIndex) {
    
    // Create result table
    auto resultTable = std::make_shared<Table>("join_result");
    
    // Add columns from left table
    for (const auto& colName : leftTable->getColumnNames()) {
        int colIdx = leftTable->getColumnIndex(colName);
        resultTable->addColumn(colName, leftTable->isColumnPrimaryKey(colIdx), leftTable->isColumnForeignKey(colIdx));
    }
    
    // Add columns from right table (skip join column to avoid duplication)
    for (const auto& colName : rightTable->getColumnNames()) {
        // Skip columns with the same name as in the left table
        if (leftTable->getColumnIndex(colName) >= 0) {
            continue;
        }
        
        int colIdx = rightTable->getColumnIndex(colName);
        resultTable->addColumn(colName, rightTable->isColumnPrimaryKey(colIdx), rightTable->isColumnForeignKey(colIdx));
    }
    
    // Build hash table for right table
    std::unordered_multimap<std::string, size_t> hashTable;
    for (size_t i = 0; i < rightTable->getRowCount(); ++i) {
        hashTable.emplace(rightTable->getValue(i, rightColumnIndex), i);
    }
    
    // Perform join
    for (size_t i = 0; i < leftTable->getRowCount(); ++i) {
        const std::string& leftValue = leftTable->getValue(i, leftColumnIndex);
        
        auto range = hashTable.equal_range(leftValue);
        for (auto it = range.first; it != range.second; ++it) {
            size_t rightRowIndex = it->second;
            
            // Create new row by combining left and right
            std::vector<std::string> newRow;
            
            // Add values from left table
            for (size_t j = 0; j < leftTable->getColumnCount(); ++j) {
                newRow.push_back(leftTable->getValue(i, j));
            }
            
            // Add values from right table (skip join column)
            for (size_t j = 0; j < rightTable->getColumnCount(); ++j) {
                const std::string& rightColName = rightTable->getColumnNames()[j];
                
                // Skip columns with the same name as in the left table
                if (leftTable->getColumnIndex(rightColName) >= 0) {
                    continue;
                }
                
                newRow.push_back(rightTable->getValue(rightRowIndex, j));
            }
            
            resultTable->addRow(newRow);
        }
    }
    
    Logger::debug("CPU inner join completed, result has " + 
                 std::to_string(resultTable->getRowCount()) + " rows");
    
    return resultTable;
}

std::shared_ptr<Table> Join::innerJoinGPU(
    const std::shared_ptr<Table>& leftTable,
    const std::shared_ptr<Table>& rightTable,
    int leftColumnIndex,
    int rightColumnIndex) {
    
    // For small tables, CPU join might be faster
    if (leftTable->getRowCount() < 1000 || rightTable->getRowCount() < 1000) {
        Logger::debug("Tables too small for GPU join, falling back to CPU");
        return innerJoinCPU(leftTable, rightTable, leftColumnIndex, rightColumnIndex);
    }
    
    Logger::debug("Starting GPU inner join operation");
    
    // Create result table with the same schema as the CPU version
    auto resultTable = std::make_shared<Table>("join_result");
    
    // Add columns from left table
    for (const auto& colName : leftTable->getColumnNames()) {
        int colIdx = leftTable->getColumnIndex(colName);
        resultTable->addColumn(colName, leftTable->isColumnPrimaryKey(colIdx), leftTable->isColumnForeignKey(colIdx));
    }
    
    // Add columns from right table (skip duplicate columns)
    for (const auto& colName : rightTable->getColumnNames()) {
        // Skip columns with the same name as in the left table
        if (leftTable->getColumnIndex(colName) >= 0) {
            continue;
        }
        
        int colIdx = rightTable->getColumnIndex(colName);
        resultTable->addColumn(colName, rightTable->isColumnPrimaryKey(colIdx), rightTable->isColumnForeignKey(colIdx));
    }
    
    // Get table sizes
    size_t leftNumRows = leftTable->getRowCount();
    size_t rightNumRows = rightTable->getRowCount();
    size_t leftNumCols = leftTable->getColumnCount();
    size_t rightNumCols = rightTable->getColumnCount();
    
    // For now, implement a CPU hash join as a fallback
    // In a real implementation, this would be replaced with CUDA kernel calls
    
    // Build hash table for right table
    std::unordered_multimap<std::string, size_t> hashTable;
    for (size_t i = 0; i < rightNumRows; ++i) {
        hashTable.emplace(rightTable->getValue(i, rightColumnIndex), i);
    }
    
    // Perform join using the hash table
    for (size_t i = 0; i < leftNumRows; ++i) {
        const std::string& leftValue = leftTable->getValue(i, leftColumnIndex);
        
        auto range = hashTable.equal_range(leftValue);
        for (auto it = range.first; it != range.second; ++it) {
            size_t rightRowIndex = it->second;
            
            // Create new row by combining left and right
            std::vector<std::string> newRow;
            
            // Add values from left table
            for (size_t j = 0; j < leftNumCols; ++j) {
                newRow.push_back(leftTable->getValue(i, j));
            }
            
            // Add values from right table (skip join column)
            for (size_t j = 0; j < rightNumCols; ++j) {
                const std::string& rightColName = rightTable->getColumnNames()[j];
                
                // Skip columns with the same name as in the left table
                if (leftTable->getColumnIndex(rightColName) >= 0) {
                    continue;
                }
                
                newRow.push_back(rightTable->getValue(rightRowIndex, j));
            }
            
            resultTable->addRow(newRow);
        }
    }
    
    Logger::debug("GPU inner join completed (CPU fallback), result has " + 
                 std::to_string(resultTable->getRowCount()) + " rows");
    
    return resultTable;
}

// CUDA kernels would be defined in a separate .cu file:
/*
__global__ void buildHashTableKernel(const char** tableData, int* hashTable, int* hashNext, 
                                   int columnIndex, int numRows, int hashSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRows) {
        // Hash the value and insert into hash table
        const char* value = tableData[idx * numCols + columnIndex];
        unsigned int hash = hashFunction(value) % hashSize;
        
        // Use atomic operations for thread safety
        int old = atomicExch(&hashNext[idx], -1);
        old = atomicExch(&hashTable[hash], idx);
        if (old != -1) {
            atomicExch(&hashNext[idx], old);
        }
    }
}

__global__ void probeHashJoinKernel(const char** leftData, const char** rightData,
                                   int* hashTable, int* hashNext, char** outputData,
                                   int leftColumnIndex, int rightColumnIndex,
                                   int leftNumColumns, int rightNumColumns,
                                   int leftNumRows, int rightNumRows,
                                   int hashSize, int* outputRowCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < leftNumRows) {
        // Get the value to join on
        const char* value = leftData[idx * leftNumColumns + leftColumnIndex];
        
        // Hash the value
        unsigned int hash = hashFunction(value) % hashSize;
        
        // Probe the hash table
        int rightIdx = hashTable[hash];
        while (rightIdx != -1) {
            // Check if values match
            const char* rightValue = rightData[rightIdx * rightNumColumns + rightColumnIndex];
            if (strcmp(value, rightValue) == 0) {
                // Values match, add this pair to the result
                int resultIdx = atomicAdd(outputRowCount, 1);
                
                // Copy left row values
                for (int i = 0; i < leftNumColumns; i++) {
                    char* dst = outputData[resultIdx * (leftNumColumns + rightNumColumns - 1) + i];
                    const char* src = leftData[idx * leftNumColumns + i];
                    strcpy(dst, src);
                }
                
                // Copy right row values (skip join column)
                int resultColIdx = leftNumColumns;
                for (int i = 0; i < rightNumColumns; i++) {
                    if (i != rightColumnIndex) {
                        char* dst = outputData[resultIdx * (leftNumColumns + rightNumColumns - 1) + resultColIdx];
                        const char* src = rightData[rightIdx * rightNumColumns + i];
                        strcpy(dst, src);
                        resultColIdx++;
                    }
                }
            }
            
            // Move to next item in hash chain
            rightIdx = hashNext[rightIdx];
        }
    }
}
*/

} // namespace SQLQueryProcessor