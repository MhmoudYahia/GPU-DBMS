#ifndef AGGREGATOR_GPU_CUH
#define AGGREGATOR_GPU_CUH

#include <cuda_runtime.h>
#include "../../include/DataHandling/Condition.hpp"
#include "../../include/DataHandling/Table.hpp"
#include "../../include/Utilities/GPU.cuh"

#include <cfloat>
#include <unordered_set>

/**
 * @enum AggregateFunction
 * @brief Supported aggregate functions
 */
enum class AggregateFunction
{
    COUNT,
    SUM,
    AVG,
    MIN,
    MAX
};

struct AggregationInfo
{
    AggregateFunction type;
    GPUDBMS::DataType dataType;
    void *inputData;
    void *outputData;
    size_t outputSize;
};

struct GroupByInfo
{
    GPUDBMS::DataType keyType;
    void *keyData;
    int numGroups;
    int *groupIndices;
};

__global__ void aggregationKernel(
    int numRows,
    int numAggregations,
    const AggregationInfo *aggregations,
    const GroupByInfo *groupBy,
    bool *filterFlags = nullptr);

extern "C" GPUDBMS::Table launchAggregationKernel(
    const GPUDBMS::Table &inputTable,
    const std::vector<std::string> &groupByColumns,
    const std::vector<std::pair<std::string, AggregateFunction>> &aggregations,
    const std::vector<GPUDBMS::Condition> &conditions = {});

#endif