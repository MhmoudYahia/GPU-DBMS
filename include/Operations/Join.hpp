#pragma once
#include "DataHandling/Table.hpp"
#include <memory>
#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace SQLQueryProcessor {

enum class JoinType {
    INNER,
    LEFT,
    RIGHT,
    FULL
};

class Join {
public:
    Join() = default;
    ~Join() = default;
    
    // CPU implementation of join operation
    std::shared_ptr<Table> executeCPU(
        const std::shared_ptr<Table>& leftTable,
        const std::shared_ptr<Table>& rightTable,
        const std::string& leftColumn,
        const std::string& rightColumn,
        JoinType joinType = JoinType::INNER);
        
    // GPU implementation of join operation
    std::shared_ptr<Table> executeGPU(
        const std::shared_ptr<Table>& leftTable,
        const std::shared_ptr<Table>& rightTable,
        const std::string& leftColumn,
        const std::string& rightColumn,
        JoinType joinType = JoinType::INNER);
        
private:
    // Helper methods for different join types
    std::shared_ptr<Table> innerJoinCPU(
        const std::shared_ptr<Table>& leftTable,
        const std::shared_ptr<Table>& rightTable,
        int leftColumnIndex,
        int rightColumnIndex);
        
    std::shared_ptr<Table> innerJoinGPU(
        const std::shared_ptr<Table>& leftTable,
        const std::shared_ptr<Table>& rightTable,
        int leftColumnIndex,
        int rightColumnIndex);
};

// CUDA kernel for hash table creation
__global__ void buildHashTableKernel(const char** tableData, int* hashTable, int* hashNext,
                                     int columnIndex, int numRows, int hashSize);

// CUDA kernel for probing hash table and joining
__global__ void probeHashJoinKernel(const char** leftData, const char** rightData,
                                   int* hashTable, int* hashNext, char** outputData,
                                   int leftColumnIndex, int rightColumnIndex,
                                   int leftNumColumns, int rightNumColumns,
                                   int leftNumRows, int rightNumRows,
                                   int hashSize, int* outputRowCount);

} // namespace SQLQueryProcessor