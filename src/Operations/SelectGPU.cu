#include "../../include/Operations/SelectGPU.cuh"
#include <iostream>

__global__ void selectKernel(
    GPUDBMS::ComparisonOperator op,
    int numRows,
    const int *intCol,
    const float *floatCol,
    const bool *boolCol,
    const double *doubleCol,
    const char *stringBuffer,
    const int *stringOffsets,
    const GPUDBMS::DataType columnType,
    bool *outputFlags,
    int intQuery,
    float floatQuery,
    double doubleQuery,
    bool boolQuery,
    const char *stringQuery)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows)
        return;

    bool result = false;

    switch (columnType)
    {
    case GPUDBMS::DataType::INT:
        result = compare(op, intCol[row], intQuery); // Pass parsed value instead of parsing in kernel
        break;
    case GPUDBMS::DataType::FLOAT:
        result = compare(op, floatCol[row], floatQuery);
        break;
    case GPUDBMS::DataType::DOUBLE:
        result = compare(op, doubleCol[row], doubleQuery);
        break;
    case GPUDBMS::DataType::BOOL:
        result = compare(op, boolCol[row], boolQuery);
        break;
    // case GPUDBMS::DataType::STRING:
    // case GPUDBMS::DataType::VARCHAR:
    // {
    //     const char *str = &stringBuffer[stringOffsets[row * 256]];
    //     // result = compareString(op, str, stringQuery);
    // }
    // break;
    default:
        result = false;
        break;
    }

    outputFlags[row] = result;
}

__device__ bool compare(GPUDBMS::ComparisonOperator op, int a, int b)
{
    switch (op)
    {
    case GPUDBMS::ComparisonOperator::EQUAL:
        return a == b;
    case GPUDBMS::ComparisonOperator::NOT_EQUAL:
        return a != b;
    case GPUDBMS::ComparisonOperator::LESS_THAN:
        return a < b;
    case GPUDBMS::ComparisonOperator::GREATER_THAN:
        return a > b;
    case GPUDBMS::ComparisonOperator::LESS_EQUAL:
        return a <= b;
    case GPUDBMS::ComparisonOperator::GREATER_EQUAL:
        return a >= b;
    default:
        return false;
    }
}

__device__ bool compare(GPUDBMS::ComparisonOperator op, float a, float b)
{
    return compare(op, (double)a, (double)b);
}

__device__ bool compare(GPUDBMS::ComparisonOperator op, double a, double b)
{
    switch (op)
    {
    case GPUDBMS::ComparisonOperator::EQUAL:
        return fabs(a - b) < 1e-6;
    case GPUDBMS::ComparisonOperator::NOT_EQUAL:
        return fabs(a - b) >= 1e-6;
    case GPUDBMS::ComparisonOperator::LESS_THAN:
        return a < b;
    case GPUDBMS::ComparisonOperator::GREATER_THAN:
        return a > b;
    case GPUDBMS::ComparisonOperator::LESS_EQUAL:
        return a <= b;
    case GPUDBMS::ComparisonOperator::GREATER_EQUAL:
        return a >= b;
    default:
        return false;
    }
}

// __device__ bool compareString(GPUDBMS::ComparisonOperator op, const char *a, const char *b)
// {
//     int cmp = strcmp(a, b);
//     switch (op)
//     {
//     case GPUDBMS::ComparisonOperator::EQUAL:
//         return cmp == 0;
//     case GPUDBMS::ComparisonOperator::NOT_EQUAL:
//         return cmp != 0;
//     case GPUDBMS::ComparisonOperator::LESS_THAN:
//         return cmp < 0;
//     case GPUDBMS::ComparisonOperator::GREATER_THAN:
//         return cmp > 0;
//     case GPUDBMS::ComparisonOperator::LESS_EQUAL:
//         return cmp <= 0;
//     case GPUDBMS::ComparisonOperator::GREATER_EQUAL:
//         return cmp >= 0;
//     default:
//         return false;
//     }
// }

extern "C" GPUDBMS::Table launchSelectKernel(
    const GPUDBMS::Table &m_inputTable,
    const GPUDBMS::Condition &m_condition)
{
    GPUDBMS::Table resultTable = m_inputTable.createEmptyWithSameSchema();
    const size_t rowCount = m_inputTable.getRowCount();
    const size_t colCount = m_inputTable.getColumnCount();

    std::unordered_map<std::string, int> columnNameToIndex;
    for (size_t i = 0; i < colCount; ++i)
    {
        columnNameToIndex[m_inputTable.getColumns()[i].getName()] = static_cast<int>(i);
    }

    std::vector<GPUDBMS::DataType> colsType = m_inputTable.getColumnsType();

    std::string cudaOperation = m_condition.getCUDACondition();

    // Remove parentheses
    cudaOperation = cudaOperation.substr(1, cudaOperation.length() - 2);

    size_t firstSpacePos = cudaOperation.find(' ');
    std::string columnName = cudaOperation.substr(0, firstSpacePos);

    // Find the second space to get the operator
    size_t secondSpacePos = cudaOperation.find(' ', firstSpacePos + 1);
    std::string opStr = cudaOperation.substr(firstSpacePos + 1, secondSpacePos - firstSpacePos - 1);

    // Get the value (rest of the string)
    std::string value = cudaOperation.substr(secondSpacePos + 1);

    // Find the column index
    if (columnNameToIndex.find(columnName) == columnNameToIndex.end())
    {
        // Column not found
        return resultTable;
    }
    int columnIndex = columnNameToIndex[columnName];
    GPUDBMS::DataType columnType = colsType[columnIndex];

    // Parse the operator
    GPUDBMS::ComparisonOperator op;
    if (opStr == "==")
        op = GPUDBMS::ComparisonOperator::EQUAL;
    else if (opStr == "!=")
        op = GPUDBMS::ComparisonOperator::NOT_EQUAL;
    else if (opStr == "<")
        op = GPUDBMS::ComparisonOperator::LESS_THAN;
    else if (opStr == ">")
        op = GPUDBMS::ComparisonOperator::GREATER_THAN;
    else if (opStr == "<=")
        op = GPUDBMS::ComparisonOperator::LESS_EQUAL;
    else if (opStr == ">=")
        op = GPUDBMS::ComparisonOperator::GREATER_EQUAL;
    else
        return resultTable; // Invalid operator

    // Allocate memory for output flags
    bool *h_outputFlags;
    cudaMallocHost((void **)&h_outputFlags, rowCount * sizeof(bool));
    std::fill(h_outputFlags, h_outputFlags + rowCount, false);

    // Allocate memory for column data
    const int *h_intCol = nullptr;
    const float *h_floatCol = nullptr;
    const bool *h_boolCol = nullptr;
    const double *h_doubleCol = nullptr;
    const char *h_stringBuffer = nullptr;
    const int *h_stringOffsets = nullptr;

    const GPUDBMS::ColumnData &cd = m_inputTable.getColumnData(columnName);
    auto &intColumn = static_cast<const GPUDBMS::ColumnDataImpl<int> &>(cd);
    auto &floatColumn = static_cast<const GPUDBMS::ColumnDataImpl<float> &>(cd);
    auto &boolColumn = static_cast<const GPUDBMS::ColumnDataImpl<bool> &>(cd);
    auto &doubleColumn = static_cast<const GPUDBMS::ColumnDataImpl<double> &>(cd);
    // auto &stringColumn = static_cast<const GPUDBMS::ColumnDataImpl<std::string> &>(cd);
    auto &stringBufferColumn = static_cast<const GPUDBMS::ColumnDataImpl<char> &>(cd);
    auto &stringOffsetsColumn = static_cast<const GPUDBMS::ColumnDataImpl<int> &>(cd);
    // Create a temporary container for bools
    std::vector<char> boolTempContainer;
    switch (columnType)
    {
    case GPUDBMS::DataType::INT:
        h_intCol = intColumn.getData().data();
        break;
    case GPUDBMS::DataType::FLOAT:
        h_floatCol = floatColumn.getData().data();
    case GPUDBMS::DataType::BOOL:
    {
        // Copy bool vector to char vector (since std::vector<bool> doesn't have proper data() method)
        const auto &boolData = boolColumn.getData();
        boolTempContainer.resize(rowCount);
        for (size_t i = 0; i < rowCount; ++i)
        {
            boolTempContainer[i] = boolData[i] ? 1 : 0;
        }
        h_boolCol = reinterpret_cast<const bool *>(boolTempContainer.data());
    }
    break;
        break;
    case GPUDBMS::DataType::DOUBLE:
        h_doubleCol = doubleColumn.getData().data();
        break;
    case GPUDBMS::DataType::STRING:
    case GPUDBMS::DataType::VARCHAR:
        h_stringBuffer = stringBufferColumn.getData().data();
        h_stringOffsets = stringOffsetsColumn.getData().data();
        break;
    default:
        // Unsupported type
        return resultTable;
    }

    // Allocate device memory for the column data
    int *d_intCol = nullptr;
    float *d_floatCol = nullptr;
    bool *d_boolCol = nullptr;
    double *d_doubleCol = nullptr;
    char *d_stringBuffer = nullptr;
    int *d_stringOffsets = nullptr;
    bool *d_outputFlags = nullptr;

    switch (columnType)
    {
    case GPUDBMS::DataType::INT:
        cudaMalloc((void **)&d_intCol, rowCount * sizeof(int));
        cudaMemcpy(d_intCol, h_intCol, rowCount * sizeof(int), cudaMemcpyHostToDevice);
        break;
    case GPUDBMS::DataType::FLOAT:
        cudaMalloc((void **)&d_floatCol, rowCount * sizeof(float));
        cudaMemcpy(d_floatCol, h_floatCol, rowCount * sizeof(float), cudaMemcpyHostToDevice);
        break;
    case GPUDBMS::DataType::BOOL:
        cudaMalloc((void **)&d_boolCol, rowCount * sizeof(bool));
        cudaMemcpy(d_boolCol, h_boolCol, rowCount * sizeof(bool), cudaMemcpyHostToDevice);
        break;
    case GPUDBMS::DataType::DOUBLE:
        cudaMalloc((void **)&d_doubleCol, rowCount * sizeof(double));
        cudaMemcpy(d_doubleCol, h_doubleCol, rowCount * sizeof(double), cudaMemcpyHostToDevice);
        break;
    case GPUDBMS::DataType::STRING:
    case GPUDBMS::DataType::VARCHAR:
        cudaMalloc((void **)&d_stringBuffer, rowCount * 256 * sizeof(char));
        cudaMemcpy(d_stringBuffer, h_stringBuffer, rowCount * 256 * sizeof(char), cudaMemcpyHostToDevice);
        cudaMalloc((void **)&d_stringOffsets, rowCount * sizeof(int));
        cudaMemcpy(d_stringOffsets, h_stringOffsets, rowCount * sizeof(int), cudaMemcpyHostToDevice);
        break;

    default:
        break;
    }

    cudaMalloc((void **)&d_outputFlags, rowCount * sizeof(bool));
    cudaMemcpy(d_outputFlags, h_outputFlags, rowCount * sizeof(bool), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 1024;
    int numBlocks = (rowCount + blockSize - 1) / blockSize;

    std::cout << "Launching kernel with " << numBlocks << " blocks and " << blockSize << " threads per block." << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    selectKernel<<<numBlocks, blockSize>>>(
        op,
        rowCount,
        d_intCol,
        d_floatCol,
        d_boolCol,
        d_doubleCol,
        d_stringBuffer,
        d_stringOffsets,
        columnType,
        d_outputFlags,
        std::stoi(value),                  // Pass parsed value as int
        std::stof(value),                  // Pass parsed value as float
        std::stod(value),                  // Pass parsed value as double
        (value == "true" || value == "1"), // Pass parsed value as bool
        value.c_str()                      // Pass parsed value as string
    );
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceSynchronize();
    // Copy the output flags back to host
    cudaMemcpy(h_outputFlags, d_outputFlags, rowCount * sizeof(bool), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_intCol);
    cudaFree(d_floatCol);
    cudaFree(d_boolCol);
    cudaFree(d_doubleCol);
    cudaFree(d_stringBuffer);
    cudaFree(d_stringOffsets);
    cudaFree(d_outputFlags);

    const auto &columns = m_inputTable.getColumns();

    std::vector<GPUDBMS::DataType> columnTypes(colCount);
    for (size_t col = 0; col < colCount; ++col)
    {
        columnTypes[col] = columns[col].getType();
    }

    std::vector<int> includedRows;
    // Fill the result table with rows that match the condition
    for (size_t row = 0; row < rowCount; ++row)
    {
        // std::cout << "Row " << row << ": " << h_outputFlags[row] << std::endl;
        if (h_outputFlags[row])
        {
            includedRows.push_back(static_cast<int>(row));
        }
    }

    return m_inputTable.getSlicedTable(includedRows);
}