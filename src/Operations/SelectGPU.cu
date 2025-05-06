#include "../../include/Operations/SelectGPU.cuh"
#include <iostream>


__global__ void selectKernel(
    int numRows,
    int numConditions,
    const ConditionGPU *conditions,
    bool *outputFlags)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows)
        return;

    bool finalResult = false;

    for (int i = 0; i < numConditions; i++)
    {
        const ConditionGPU &cond = conditions[i];
        bool currentResult = false;

        // Evaluate the current condition
        switch (cond.columnInfo.type)
        {
        case GPUDBMS::DataType::INT:
        {
            // printf("INT\n");
            const int *col = static_cast<const int *>(cond.columnInfo.data);
            const int query = *static_cast<const int *>(cond.queryValue);
            // printf("Comparing %d with %d\n", col[row], query);
            currentResult = compare(cond.comparisonOp, col[row], query);
            break;
        }
        case GPUDBMS::DataType::FLOAT:
        {
            const float *col = static_cast<const float *>(cond.columnInfo.data);
            const float query = *static_cast<const float *>(cond.queryValue);
            // printf("Comparing %f with %f\n", col[row], query);
            currentResult = compare(cond.comparisonOp, col[row], query);
            break;
        }
        case GPUDBMS::DataType::DOUBLE:
        {
            const double *col = static_cast<const double *>(cond.columnInfo.data);
            const double query = *static_cast<const double *>(cond.queryValue);
            // printf("Comparing %f with %f\n", col[row], query);
            currentResult = compare(cond.comparisonOp, col[row], query);
            break;
        }
        case GPUDBMS::DataType::BOOL:
        {
            const bool *col = static_cast<const bool *>(cond.columnInfo.data);
            const bool query = *static_cast<const bool *>(cond.queryValue);
            // printf("Comparing %d with %d\n", col[row], query);
            currentResult = compare(cond.comparisonOp, col[row], query);
            break;
        }
            // Add string cases if needed
        }

        // Apply logical operator to combine with previous result
        if (i == 0)
        {
            // First condition - just take its result
            finalResult = currentResult;
        }
        else
        {
            switch (cond.logicalOp)
            {
            case GPUDBMS::LogicalOperator::AND:
                finalResult = finalResult && currentResult;
                break;
            case GPUDBMS::LogicalOperator::OR:
                finalResult = finalResult || currentResult;
                break;
            case GPUDBMS::LogicalOperator::NOT:
                finalResult = !currentResult;
                break;
            default:
                // Default to AND if unknown operator
                finalResult = finalResult && currentResult;
            }
        }

        // Optional: Early exit optimization for certain cases
        if (finalResult && cond.logicalOp == GPUDBMS::LogicalOperator::OR)
        {
            // If we have a true result with OR, we can exit early
            break;
        }
    }

    outputFlags[row] = finalResult;
}

size_t min_size_t(size_t a, size_t b, size_t c)
{
    return std::min(std::min(a, b), c);
}

std::string cleanToken(const std::string &token)
{
    std::string trimmed = token;
    // Remove leading/trailing spaces
    trimmed.erase(0, trimmed.find_first_not_of(" \t\r\n"));
    trimmed.erase(trimmed.find_last_not_of(" \t\r\n") + 1);
    // Remove surrounding parentheses
    while (!trimmed.empty() && trimmed.front() == '(' && trimmed.back() == ')')
    {
        trimmed = trimmed.substr(1, trimmed.size() - 2);
        trimmed.erase(0, trimmed.find_first_not_of(" \t\r\n"));
        trimmed.erase(trimmed.find_last_not_of(" \t\r\n") + 1);
    }
    return trimmed;
}

// Helper function to parse conditions string into Condition structs
std::vector<ConditionGPU> parseConditions(const GPUDBMS::Table &m_inputTable, const std::string &conditionStr,
                                          const std::unordered_map<std::string, int> &columnNameToIndex, const std::vector<GPUDBMS::DataType> &colsType)
{
    std::vector<ConditionGPU> conditions;

    // Remove outer parentheses if present
    std::string processedStr = conditionStr;
    if (processedStr.front() == '(' && processedStr.back() == ')')
        processedStr = processedStr.substr(1, processedStr.length() - 2);

    // Split by logical operators
    size_t pos = 0;
    std::string token;
    GPUDBMS::LogicalOperator currentLogicalOp = GPUDBMS::LogicalOperator::NONE;

    while (pos < processedStr.length())
    {
        // Check for logical operators
        if (pos + 4 <= processedStr.length() && processedStr.substr(pos, 4) == " && ")
        {
            currentLogicalOp = GPUDBMS::LogicalOperator::AND;
            pos += 4;
            continue;
        }
        else if (pos + 4 <= processedStr.length() && processedStr.substr(pos, 4) == " || ")
        {
            currentLogicalOp = GPUDBMS::LogicalOperator::OR;
            pos += 4;
            continue;
        }
        else if (pos + 3 <= processedStr.length() && processedStr.substr(pos, 3) == " ! ")
        {
            currentLogicalOp = GPUDBMS::LogicalOperator::NOT;
            pos += 3;
            continue;
        }

        // Find the end of the current condition
        size_t nextAndPos = processedStr.find(" && ", pos);
        size_t nextOrPos = processedStr.find(" || ", pos);
        size_t nextNotPos = processedStr.find(" ! ", pos);
        size_t endPos = min_size_t(
            nextAndPos == std::string::npos ? processedStr.length() : nextAndPos,
            nextOrPos == std::string::npos ? processedStr.length() : nextOrPos,
            nextNotPos == std::string::npos ? processedStr.length() : nextNotPos);

        // Extract the condition
        std::string conditionToken = cleanToken(processedStr.substr(pos, endPos - pos));
        pos = endPos;

        // Parse individual condition (column op value)
        size_t firstSpacePos = conditionToken.find(' ');
        if (firstSpacePos == std::string::npos)
            continue;

        std::string columnName = conditionToken.substr(0, firstSpacePos);
        // Trim leading/trailing spaces and parentheses
        columnName.erase(0, columnName.find_first_not_of(" ("));
        columnName.erase(columnName.find_last_not_of(" )") + 1);

        // Find the operator
        size_t secondSpacePos = conditionToken.find(' ', firstSpacePos + 1);
        if (secondSpacePos == std::string::npos)
            continue;

        std::string opStr = conditionToken.substr(firstSpacePos + 1, secondSpacePos - firstSpacePos - 1);

        // Get the value
        std::string valueStr = conditionToken.substr(secondSpacePos + 1);

        // Remove quotes from string values
        if (valueStr.front() == '\'' && valueStr.back() == '\'')
        {
            valueStr = valueStr.substr(1, valueStr.length() - 2);
        }

        // Check if the column exists
        if (columnNameToIndex.find(columnName) == columnNameToIndex.end())
        {
            std::cerr << "Column not found: " << columnName << std::endl;
            continue;
        }

        // Get the column type
        int columnIndex = columnNameToIndex.at(columnName);

        // Parse comparison operator
        GPUDBMS::ComparisonOperator compOp;
        if (opStr == "==")
            compOp = GPUDBMS::ComparisonOperator::EQUAL;
        else if (opStr == "!=")
            compOp = GPUDBMS::ComparisonOperator::NOT_EQUAL;
        else if (opStr == "<")
            compOp = GPUDBMS::ComparisonOperator::LESS_THAN;
        else if (opStr == ">")
            compOp = GPUDBMS::ComparisonOperator::GREATER_THAN;
        else if (opStr == "<=")
            compOp = GPUDBMS::ComparisonOperator::LESS_EQUAL;
        else if (opStr == ">=")
            compOp = GPUDBMS::ComparisonOperator::GREATER_EQUAL;
        else
        {
            std::cerr << "Invalid operator: " << opStr << std::endl;
            continue;
        }

        // Create and populate condition
        ConditionGPU condition;
        condition.comparisonOp = compOp;
        condition.logicalOp = currentLogicalOp;
        condition.columnInfo.name = columnName;
        condition.columnInfo.type = colsType[columnIndex];

        // Handle different data types
        switch (colsType[columnIndex])
        {
        case GPUDBMS::DataType::INT:
        {
            int *value = new int(std::stoi(valueStr));
            condition.queryValue = static_cast<void *>(value);
            break;
        }
        case GPUDBMS::DataType::FLOAT:
        {
            float *value = new float(std::stof(valueStr));
            condition.queryValue = static_cast<void *>(value);
            break;
        }
        case GPUDBMS::DataType::DOUBLE:
        {
            double *value = new double(std::stod(valueStr));
            condition.queryValue = static_cast<void *>(value);
            break;
        }
        case GPUDBMS::DataType::BOOL:
        {
            bool *value = new bool(valueStr == "true" || valueStr == "1");
            condition.queryValue = static_cast<void *>(value);
            break;
        }
        case GPUDBMS::DataType::STRING:
        case GPUDBMS::DataType::VARCHAR:
        {
            // Allocate memory for the string value (up to 256 chars)
            char *buffer = new char[256];
            strncpy(buffer, valueStr.c_str(), 255);
            buffer[255] = '\0';
            condition.queryValue = static_cast<void *>(buffer);
            break;
        }
        default:
            std::cerr << "Unsupported data type for column: " << columnName << std::endl;
            continue;
        }

        condition.columnInfo=m_inputTable.getColumnInfoGPU(columnName);
       

        conditions.push_back(condition);
    }

    return conditions;
}

extern "C" GPUDBMS::Table launchSelectKernel(
    const GPUDBMS::Table &m_inputTable,
    const GPUDBMS::Condition &m_condition)
{
    GPUDBMS::Table resultTable = m_inputTable.createEmptyWithSameSchema();
    const size_t rowCount = m_inputTable.getRowCount();
    const size_t colCount = m_inputTable.getColumnCount();

    // Create column name to index mapping
    std::unordered_map<std::string, int> columnNameToIndex;
    for (size_t i = 0; i < colCount; ++i)
    {
        columnNameToIndex[m_inputTable.getColumns()[i].getName()] = static_cast<int>(i);
    }

    std::vector<GPUDBMS::DataType> colsType = m_inputTable.getColumnsType();

    // Parse the condition into individual components
    std::vector<ConditionGPU>
        conditions = parseConditions(m_inputTable, m_condition.getCUDACondition(), columnNameToIndex, colsType);
    if (conditions.empty())
    {
        return resultTable;
    }

    // Debug
    for (auto &cond : conditions)
    {
        std::cout << "Condition: " << cond.columnInfo.name << " ";
        switch (cond.comparisonOp)
        {
        case GPUDBMS::ComparisonOperator::EQUAL:
            std::cout << "==";
            break;
        case GPUDBMS::ComparisonOperator::NOT_EQUAL:
            std::cout << "!=";
            break;
        case GPUDBMS::ComparisonOperator::LESS_THAN:
            std::cout << "<";
            break;
        case GPUDBMS::ComparisonOperator::GREATER_THAN:
            std::cout << ">";
            break;
        case GPUDBMS::ComparisonOperator::LESS_EQUAL:
            std::cout << "<=";
            break;
        case GPUDBMS::ComparisonOperator::GREATER_EQUAL:
            std::cout << ">=";
            break;
        }
        std::cout << " ";

        switch (cond.columnInfo.type)
        {
        case GPUDBMS::DataType::INT:
            std::cout << *static_cast<const int *>(cond.queryValue);
            break;
        case GPUDBMS::DataType::FLOAT:
            std::cout << *static_cast<const float *>(cond.queryValue);
            break;
        case GPUDBMS::DataType::DOUBLE:
            std::cout << *static_cast<const double *>(cond.queryValue);
            break;
        case GPUDBMS::DataType::BOOL:
            std::cout << (*static_cast<const bool *>(cond.queryValue) ? "true" : "false");
            break;
        case GPUDBMS::DataType::STRING:
        case GPUDBMS::DataType::VARCHAR:
            std::cout << static_cast<const char *>(cond.queryValue);
            break;
        default:
            std::cout << "[unknown type]";
        }
        std::cout << std::endl;
    }

    // Allocate memory for output flags
    bool *h_outputFlags;
    cudaMallocHost((void **)&h_outputFlags, rowCount * sizeof(bool));
    std::fill(h_outputFlags, h_outputFlags + rowCount, false);

    // Prepare device memory for conditions and their data
    ConditionGPU *d_conditions = nullptr;
    cudaMalloc((void **)&d_conditions, conditions.size() * sizeof(ConditionGPU));

    // Create a copy of conditions that we'll modify to point to device memory
    std::vector<ConditionGPU> device_conditions = conditions;

    // Allocate memory for each condition's data and query value
    for (size_t i = 0; i < conditions.size(); i++)
    {
        // Allocate and copy column data
        void *d_columnData = nullptr;
        size_t dataSize = getTypeSize(conditions[i].columnInfo.type) * rowCount;
        cudaMalloc(&d_columnData, dataSize);
        cudaMemcpy(d_columnData, conditions[i].columnInfo.data, dataSize, cudaMemcpyHostToDevice);
        device_conditions[i].columnInfo.data = d_columnData;

        // Allocate and copy query value
        void *d_queryValue = nullptr;
        size_t valueSize = getTypeSize(conditions[i].columnInfo.type);
        cudaMalloc(&d_queryValue, valueSize);
        cudaMemcpy(d_queryValue, conditions[i].queryValue, valueSize, cudaMemcpyHostToDevice);
        device_conditions[i].queryValue = d_queryValue;
    }

    // Copy the modified conditions to device
    cudaMemcpy(d_conditions, device_conditions.data(), conditions.size() * sizeof(ConditionGPU), cudaMemcpyHostToDevice);

    // Allocate device memory for output flags
    bool *d_outputFlags;
    cudaMalloc((void **)&d_outputFlags, rowCount * sizeof(bool));
    cudaMemcpy(d_outputFlags, h_outputFlags, rowCount * sizeof(bool), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 1024;
    int numBlocks = (rowCount + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    selectKernel<<<numBlocks, blockSize>>>(
        rowCount,
        conditions.size(),
        d_conditions,
        // d_columnInfos,
        // d_outputFlags,
        // d_queryValues.data()
        d_outputFlags);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceSynchronize();

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    //     return resultTable;
    // }

    // Copy the output flags back to host
    cudaMemcpy(h_outputFlags, d_outputFlags, rowCount * sizeof(bool), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_conditions);
    cudaFree(d_outputFlags);

    // Collect rows that match the condition
    std::vector<int> includedRows;
    for (size_t row = 0; row < rowCount; ++row)
    {
        if (h_outputFlags[row])
        {
            includedRows.push_back(static_cast<int>(row));
        }
    }

    cudaFreeHost(h_outputFlags);

    return m_inputTable.getSlicedTable(includedRows);
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
