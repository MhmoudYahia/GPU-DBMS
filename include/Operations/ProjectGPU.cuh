#ifndef PROJECT_GPU_CUH
#define PROJECT_GPU_CUH
#include <cuda_runtime.h>
#include "../../include/DataHandling/Condition.hpp"
#include "../../include/DataHandling/Table.hpp"
#include "../../include/Utilities/GPU.cuh"

__global__ void projectKernel(
    int numRows,
    const GPUDBMS::ColumnInfoGPU *inputColumns,
    int numInputColumns,
    const int *projectionIndices, // indices of columns to project
    int numProjectionColumns,
    GPUDBMS::ColumnInfoGPU *outputColumns);

extern "C" GPUDBMS::Table launchProjectKernel(
    const GPUDBMS::Table &inputTable,
    const std::vector<std::string> &projectColumns);

#endif