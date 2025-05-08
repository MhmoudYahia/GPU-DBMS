#ifndef JOIN_GPU_CUH
#define JOIN_GPU_CUH
#include <cuda_runtime.h>
#include "../../include/DataHandling/Condition.hpp"
#include "../../include/DataHandling/Table.hpp"
#include "../../include/Utilities/GPU.cuh"

template <typename T>
__global__ void joinKernel(
    const T* leftTable, int leftCols, int leftRows,
    const T* rightTable, int rightCols, int rightRows,
    T* outputTable, int* outputCount,
    int leftJoinCol, int rightJoinCol, int typeEnum
);

template <typename T>
void launchJoinKernel(
    Table* resultTable,int leftCols, int leftRows,
    int rightCols, int rightRows, int typeEnum
);

// extern "C" GPUDBMS::Table launchProjectKernel(
//     const GPUDBMS::Table &inputTable,
//     const std::vector<std::string> &projectColumns);

#endif