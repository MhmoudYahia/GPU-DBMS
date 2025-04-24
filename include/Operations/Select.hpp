#pragma once
#include "DataHandling/Table.hpp"
#include <memory>
#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace SQLQueryProcessor {

class Select {
public:
    Select() = default;
    ~Select() = default;
    
    // CPU implementation of select operation
    std::shared_ptr<Table> executeCPU(
        const std::shared_ptr<Table>& inputTable,
        const std::vector<std::string>& columnNames,
        const std::vector<int>& columnIndices);
        
    // GPU implementation of select operation
    std::shared_ptr<Table> executeGPU(
        const std::shared_ptr<Table>& inputTable,
        const std::vector<std::string>& columnNames,
        const std::vector<int>& columnIndices);
        
private:
    // Helper method to prepare column indices from names
    std::vector<int> prepareColumnIndices(
        const std::shared_ptr<Table>& inputTable,
        const std::vector<std::string>& columnNames);
};

// CUDA kernel for select operation
__global__ void selectKernel(const char** inputData, char** outputData, 
                            int* columnIndices, int numRows, int numColumns, 
                            int numOutputColumns);

} 