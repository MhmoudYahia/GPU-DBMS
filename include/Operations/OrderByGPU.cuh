#ifndef ORDERBY_GPU_CUH
#define ORDERBY_GPU_CUH
#include <cuda_runtime.h>
#include "../../include/DataHandling/Condition.hpp"
#include "../../include/DataHandling/Table.hpp"
#include "../../include/Utilities/GPU.cuh"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

/**
 * @enum SortOrder
 * @brief Specifies the order for sorting (ascending or descending)
 */
enum class SortOrder
{
    ASC,
    DESC
};

__global__ void generateSortKeysKernel(
    int numRows,
    const GPUDBMS::ColumnInfoGPU *sortColumns,
    int numSortColumns,
    SortOrder *sortDirections, // true=asc, false=desc
    uint64_t *keys);

void sortRows(
    uint64_t *keys,
    uint32_t *indices,
    int numRows,
    cudaStream_t stream = 0);

__global__ void reorderDataKernel(
    int numRows,
    const uint32_t *indices,
    const GPUDBMS::ColumnInfoGPU *columnsToReorder,
    int numColumns,
    void **reorderedData);

extern "C" GPUDBMS::Table launchOrderByKernel(
    const GPUDBMS::Table &inputTable,
    const std::vector<std::string> &sortColumns,
    const std::vector<SortOrder> &sortDirections);

#endif