#ifndef SELECT_GPU_CUH
#define SELECT_GPU_CUH
#include <cuda_runtime.h>
#include "../../include/DataHandling/Condition.hpp"
#include "../../include/DataHandling/Table.hpp"

__global__ void selectKernel(
    int numRows,
    int numConditions,
    const ConditionGPU *conditions,
    bool *outputFlags);

__device__ bool compare(GPUDBMS::ComparisonOperator op, int a, int b);

__device__ bool compare(GPUDBMS::ComparisonOperator op, float a, float b);

__device__ bool compare(GPUDBMS::ComparisonOperator op, double a, double b);

__device__ bool compareString(GPUDBMS::ComparisonOperator op, const char *a, const char *b);

extern "C" GPUDBMS::Table launchSelectKernel(
    const GPUDBMS::Table &m_inputTable,
    const GPUDBMS::Condition &m_condition);

#endif // SELECT_GPU_CUH