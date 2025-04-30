#include "../../include/DataHandling/ConditionGPU.cuh"

__device__ bool evaluateCondition(
    const GPUCondition &cond,
    int **intCols,
    float **floatCols,
    bool **boolCols,
    char *stringBuffer,
    int *stringOffsets,
    int rowIndex)
{
    switch (cond.type)
    {
    case GPUDBMS::DataType::INT:
    {
        int val = intCols[cond.columnIndex][rowIndex];
        switch (cond.op)
        {
        case GPUDBMS::ComparisonOperator::EQUAL:
            return val == cond.intValue;
        case GPUDBMS::ComparisonOperator::NOT_EQUAL:
            return val != cond.intValue;
        case GPUDBMS::ComparisonOperator::LESS_THAN:
            return val < cond.intValue;
        case GPUDBMS::ComparisonOperator::LESS_EQUAL:
            return val <= cond.intValue;
        case GPUDBMS::ComparisonOperator::GREATER_THAN:
            return val > cond.intValue;
        case GPUDBMS::ComparisonOperator::GREATER_EQUAL:
            return val >= cond.intValue;
        default:
            return false;
        }
    }
    case GPUDBMS::DataType::FLOAT:
    {
        float val = floatCols[cond.columnIndex][rowIndex];
        switch (cond.op)
        {
        case GPUDBMS::ComparisonOperator::EQUAL:
            return val == cond.floatValue;
        case GPUDBMS::ComparisonOperator::NOT_EQUAL:
            return val != cond.floatValue;
        case GPUDBMS::ComparisonOperator::LESS_THAN:
            return val < cond.floatValue;
        case GPUDBMS::ComparisonOperator::LESS_EQUAL:
            return val <= cond.floatValue;
        case GPUDBMS::ComparisonOperator::GREATER_THAN:
            return val > cond.floatValue;
        case GPUDBMS::ComparisonOperator::GREATER_EQUAL:
            return val >= cond.floatValue;
        default:
            return false;
        }
    }
    case GPUDBMS::DataType::BOOL:
    {
        bool val = boolCols[cond.columnIndex][rowIndex];
        if (cond.op == GPUDBMS::ComparisonOperator::EQUAL)
            return val == cond.boolValue;
        if (cond.op == GPUDBMS::ComparisonOperator::NOT_EQUAL)
            return val != cond.boolValue;
        return false;
    }
    default:
        return false; // handle STRING separately if needed
    }
};

__global__ void filterKernel(
    GPUCondition cond,
    int **intCols,
    float **floatCols,
    bool **boolCols,
    char *stringBuffer,
    int *stringOffsets,
    bool *resultFlags,
    int rowCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rowCount)
    {
        resultFlags[i] = evaluateCondition(cond, intCols, floatCols, boolCols, stringBuffer, stringOffsets, i);
    }
}

extern "C" bool *launchFilterKernel(
    std::string m_columnName,
    GPUDBMS::ComparisonOperator m_operator,
    std::string m_value,
    const std::vector<GPUDBMS::DataType> &colsType,
    const std::vector<std::string> &row,
    std::unordered_map<std::string, int> columnNameToIndex

)
{
    int numRows = row.size();
    int numCols = colsType.size();

    int **intCols = new int *[numRows];
    float **floatCols = new float *[numRows];
    bool **boolCols = new bool *[numRows];
    char *stringBuffer = new char[256 * numRows];
    int *stringOffsets = new int[numRows];
    bool *outputFlags = new bool[numRows];

    std::fill(outputFlags, outputFlags + numRows, false);

    auto it = columnNameToIndex.find(m_columnName);
    if (it == columnNameToIndex.end() || it->second >= numCols)
        return outputFlags;

    int index = it->second;

    // Prepare GPUCondition
    GPUCondition gpuCondition;
    gpuCondition.op = m_operator;
    gpuCondition.type = colsType[index];
    gpuCondition.columnIndex = index;

    switch (gpuCondition.type)
    {
    case GPUDBMS::DataType::INT:
        gpuCondition.intValue = std::stoi(m_value);
        break;
    case GPUDBMS::DataType::FLOAT:
        gpuCondition.floatValue = std::stof(m_value);
        break;
    case GPUDBMS::DataType::DOUBLE:
        gpuCondition.doubleValue = std::stod(m_value);
        break;
    case GPUDBMS::DataType::VARCHAR:
    case GPUDBMS::DataType::STRING:
        strncpy(gpuCondition.stringValue, m_value.c_str(), sizeof(gpuCondition.stringValue) - 1);
        gpuCondition.stringValue[sizeof(gpuCondition.stringValue) - 1] = '\0';
        break;
    case GPUDBMS::DataType::BOOL:
        gpuCondition.boolValue = (m_value == "true" || m_value == "1");
        break;
    default:
        return outputFlags;
    }

    for (int i = 0; i < numRows; ++i)
    {
        intCols[i] = new int[numCols];
        floatCols[i] = new float[numCols];
        boolCols[i] = new bool[numCols];

        for (int col = 0; col < numCols; ++col)
        {
            if (colsType[col] == GPUDBMS::DataType::INT)
                intCols[i][col] = std::stoi(row[col]);
            else if (colsType[col] == GPUDBMS::DataType::FLOAT)
                floatCols[i][col] = std::stof(row[col]);
            else if (colsType[col] == GPUDBMS::DataType::BOOL)
                boolCols[i][col] = (row[col] == "true" || row[col] == "1");
            else if (colsType[col] == GPUDBMS::DataType::VARCHAR || colsType[col] == GPUDBMS::DataType::STRING)
            {
                stringOffsets[i] = i * 256;
                strncpy(&stringBuffer[stringOffsets[i]], row[col].c_str(), 256);
            }
        }
    }

    // Allocate device memory
    int **d_intCols;
    float **d_floatCols;
    bool **d_boolCols;
    char *d_stringBuffer;
    int *d_stringOffsets;
    bool *d_outputFlags;
    GPUCondition *d_gpuCondition;

    cudaMalloc(&d_intCols, sizeof(int *) * numRows);
    cudaMalloc(&d_floatCols, sizeof(float *) * numRows);
    cudaMalloc(&d_boolCols, sizeof(bool *) * numRows);
    cudaMalloc(&d_stringBuffer, sizeof(char) * 256 * numRows);
    cudaMalloc(&d_stringOffsets, sizeof(int) * numRows);
    cudaMalloc(&d_outputFlags, sizeof(bool) * numRows);
    cudaMalloc(&d_gpuCondition, sizeof(GPUCondition));

    // Copy data to device
    cudaMemcpy(d_intCols, intCols, sizeof(int *) * numRows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_floatCols, floatCols, sizeof(float *) * numRows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_boolCols, boolCols, sizeof(bool *) * numRows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stringBuffer, stringBuffer, sizeof(char) * 256 * numRows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stringOffsets, stringOffsets, sizeof(int) * numRows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gpuCondition, &gpuCondition, sizeof(GPUCondition), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (numRows + blockSize - 1) / blockSize;
    filterKernel<<<numBlocks, blockSize>>>(
        *d_gpuCondition,
        d_intCols,
        d_floatCols,
        d_boolCols,
        d_stringBuffer,
        d_stringOffsets,
        d_outputFlags,
        numRows);
    
    cudaDeviceSynchronize();

    // Copy back results
    cudaMemcpy(outputFlags, d_outputFlags, sizeof(bool) * numRows, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_intCols);
    cudaFree(d_floatCols);
    cudaFree(d_boolCols);
    cudaFree(d_stringBuffer);
    cudaFree(d_stringOffsets);
    cudaFree(d_outputFlags);
    cudaFree(d_gpuCondition);

    // Free host memory
    for (int i = 0; i < numRows; ++i)
    {
        delete[] intCols[i];
        delete[] floatCols[i];
        delete[] boolCols[i];
    }
    delete[] intCols;
    delete[] floatCols;
    delete[] boolCols;
    delete[] stringBuffer;
    delete[] stringOffsets;

    return outputFlags;
}