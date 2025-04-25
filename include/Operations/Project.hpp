#pragma once
#include "DataHandling/Table.hpp"
#include <memory>
#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace SQLQueryProcessor
{

    class Project
    {
    public:
        Project() = default;
        ~Project() = default;

        // CPU implementation of project operation
        std::shared_ptr<Table> executeCPU(
            const std::shared_ptr<Table> &inputTable,
            const std::vector<std::string> &columnNames);

        // GPU implementation of project operation
        std::shared_ptr<Table> executeGPU(
            const std::shared_ptr<Table> &inputTable,
            const std::vector<std::string> &columnNames);

    private:
        // Helper method to prepare column indices from names
        std::vector<int> prepareColumnIndices(
            const std::shared_ptr<Table> &inputTable,
            const std::vector<std::string> &columnNames);
    };

    // CUDA kernel for project operation
    __global__ void projectKernel(const char **inputData, char **outputData,
                                  int *columnIndices, int numRows, int numColumns,
                                  int numOutputColumns);

} // namespace SQLQueryProcessor