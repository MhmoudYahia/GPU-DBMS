
#ifndef CONDITION_GPU_CUH
#define CONDITION_GPU_CUH

#include <cuda_runtime.h>
#include "../../include/DataHandling/Condition.hpp"
#include "../../include/DataHandling/Table.hpp"

struct GPUCondition
{
    GPUDBMS::ComparisonOperator op;
    GPUDBMS::DataType type;
    int columnIndex;
    union
    {
        int intValue;
        float floatValue;
        double doubleValue;
        bool boolValue;
        int stringOffset;
        char stringValue[256];
    };
};

__device__ bool evaluateCondition(
    const GPUCondition &cond,
    int **intCols,
    float **floatCols,
    bool **boolCols,
    char *stringBuffer,
    int *stringOffsets,
    int rowIndex);

__global__ void filterKernel(
    GPUCondition *conditions,
    int numConditions,
    int **intCols,
    float **floatCols,
    bool **boolCols,
    char *stringBuffer,
    int *stringOffsets,
    bool *outputFlags,
    int numRows);

extern "C" bool *launchFilterKernel(
    std::string m_columnName,
    GPUDBMS::ComparisonOperator m_operator,
    std::string m_value,
    const std::vector<GPUDBMS::DataType> &colsType,
    const std::vector<std::string> &row,
    std::unordered_map<std::string, int> columnNameToIndex);

#endif // CONDITION_GPU_CUH