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
            const int *col = static_cast<const int *>(cond.columnInfo.data);
            const int query = *static_cast<const int *>(cond.queryValue);
            currentResult = compare(cond.comparisonOp, col[row], query);
            break;
        }
        case GPUDBMS::DataType::FLOAT:
        {
            const float *col = static_cast<const float *>(cond.columnInfo.data);
            const float query = *static_cast<const float *>(cond.queryValue);
            currentResult = compare(cond.comparisonOp, col[row], query);
            break;
        }
        case GPUDBMS::DataType::DOUBLE:
        {
            const double *col = static_cast<const double *>(cond.columnInfo.data);
            const double query = *static_cast<const double *>(cond.queryValue);
            currentResult = compare(cond.comparisonOp, col[row], query);
            break;
        }
        case GPUDBMS::DataType::BOOL:
        {
            const bool *col = static_cast<const bool *>(cond.columnInfo.data);
            const bool query = *static_cast<const bool *>(cond.queryValue);
            currentResult = compare(cond.comparisonOp, col[row], query);
            break;
        }
        case GPUDBMS::DataType::STRING:
        case GPUDBMS::DataType::VARCHAR:
        case GPUDBMS::DataType::DATE:
        case GPUDBMS::DataType::DATETIME:
        {
            // For contiguous string storage
            const char *col = static_cast<const char *>(cond.columnInfo.data);
            const char *query = static_cast<const char *>(cond.queryValue);

            // Calculate the offset for this row's string
            const char *currentStr = col + (row * cond.columnInfo.stride);

            if (row >= cond.columnInfo.count || currentStr == nullptr || query == nullptr)
            {
                currentResult = false;
            }
            else
            {
                currentResult = compareString(cond.comparisonOp, currentStr, query);
            }
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
        else if (!finalResult && cond.logicalOp == GPUDBMS::LogicalOperator::AND)
        {
            // If we have a false result with AND, we can exit early
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
        case GPUDBMS::DataType::DATE:
        case GPUDBMS::DataType::DATETIME:
        {
            char *buffer = new char[20];
            strncpy(buffer, valueStr.c_str(), 19);
            buffer[19] = '\0'; // Null-terminate the string
            condition.queryValue = static_cast<void *>(buffer);
            break;
        }
        default:
            std::cerr << "Unsupported data type for column: " << columnName << std::endl;
            continue;
        }

        // We need to get the actual data pointer based on the column type
        const GPUDBMS::ColumnData &cd = m_inputTable.getColumnData(columnName);
        switch (colsType[columnIndex])
        {
        case GPUDBMS::DataType::INT:
        {
            auto &col = static_cast<const GPUDBMS::ColumnDataImpl<int> &>(cd);
            condition.columnInfo.data = const_cast<void *>(static_cast<const void *>(col.getData().data()));
            condition.columnInfo.count = col.getData().size();
            break;
        }
        case GPUDBMS::DataType::FLOAT:
        {
            auto &col = static_cast<const GPUDBMS::ColumnDataImpl<float> &>(cd);
            condition.columnInfo.data = const_cast<void *>(static_cast<const void *>(col.getData().data()));
            condition.columnInfo.count = col.getData().size();
            break;
        }
        case GPUDBMS::DataType::DOUBLE:
        {
            auto &col = static_cast<const GPUDBMS::ColumnDataImpl<double> &>(cd);
            condition.columnInfo.data = const_cast<void *>(static_cast<const void *>(col.getData().data()));
            condition.columnInfo.count = col.getData().size();
            break;
        }
        case GPUDBMS::DataType::BOOL:
        {
            auto &col = static_cast<const GPUDBMS::ColumnDataImpl<bool> &>(cd);
            const std::vector<bool> &boolVec = col.getData();

            // Create a host array of bools
            bool *h_boolArray = new bool[boolVec.size()];

            // Copy each bit to a full bool
            for (size_t i = 0; i < boolVec.size(); i++)
            {
                h_boolArray[i] = boolVec[i];
            }

            // We'll use this temporary host array later to copy to device
            condition.columnInfo.data = h_boolArray;
            condition.columnInfo.count = col.getData().size();

            break;
        }
        case GPUDBMS::DataType::VARCHAR:
        case GPUDBMS::DataType::STRING:
        {
            auto &col = static_cast<const GPUDBMS::ColumnDataImpl<std::string> &>(cd);
            const std::vector<std::string> &strVec = col.getData();

            // Allocate a single contiguous buffer for all strings
            const size_t maxStrLen = 256; // For VARCHAR/STRING
            char *h_contiguousBuffer = new char[strVec.size() * maxStrLen];

            // Copy strings to contiguous buffer
            for (size_t i = 0; i < strVec.size(); i++)
            {
                strncpy(h_contiguousBuffer + (i * maxStrLen),
                        strVec[i].c_str(),
                        maxStrLen - 1);
                h_contiguousBuffer[(i * maxStrLen) + maxStrLen - 1] = '\0';
            }

            condition.columnInfo.data = h_contiguousBuffer;
            condition.columnInfo.count = strVec.size();
            condition.columnInfo.stride = maxStrLen; // Add this to your ColumnInfo struct
            break;
        }
        case GPUDBMS::DataType::DATE:
        case GPUDBMS::DataType::DATETIME:
        {
            auto &col = static_cast<const GPUDBMS::ColumnDataImpl<std::string> &>(cd);
            const std::vector<std::string> &strVec = col.getData();

            // Allocate a single contiguous buffer for all strings
            const size_t maxStrLen = 20; // For DATE/DATETIME
            char *h_contiguousBuffer = new char[strVec.size() * maxStrLen];

            // Copy strings to contiguous buffer
            for (size_t i = 0; i < strVec.size(); i++)
            {
                strncpy(h_contiguousBuffer + (i * maxStrLen),
                        strVec[i].c_str(),
                        maxStrLen - 1);
                h_contiguousBuffer[(i * maxStrLen) + maxStrLen - 1] = '\0';
            }

            condition.columnInfo.data = h_contiguousBuffer;
            condition.columnInfo.count = strVec.size();
            condition.columnInfo.stride = maxStrLen; // Add this to your ColumnInfo struct
            break;
        }
            // Add cases for other data types as needed
        }

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

    for (auto &cond : conditions)
    {
        // Allocate and copy queryValue to GPU
        void *d_queryValue;
        size_t valueSize = 0;

        switch (cond.columnInfo.type)
        {
        case GPUDBMS::DataType::INT:
            valueSize = sizeof(int);
            break;
        case GPUDBMS::DataType::FLOAT:
            valueSize = sizeof(float);
            break;
        case GPUDBMS::DataType::DOUBLE:
            valueSize = sizeof(double);
            break;
        case GPUDBMS::DataType::BOOL:
            valueSize = sizeof(bool);
            break;
        case GPUDBMS::DataType::STRING:
        case GPUDBMS::DataType::VARCHAR:
            valueSize = 256;
            break; // Max string size
        case GPUDBMS::DataType::DATE:
        case GPUDBMS::DataType::DATETIME:
            valueSize = 20; // Max date/datetime size
            break;
        }

        if (cond.columnInfo.type == GPUDBMS::DataType::DATETIME || cond.columnInfo.type == GPUDBMS::DataType::DATE || cond.columnInfo.type == GPUDBMS::DataType::STRING || cond.columnInfo.type == GPUDBMS::DataType::VARCHAR)
        {

            cudaMalloc(&d_queryValue, cond.columnInfo.count * cond.columnInfo.stride);
            cudaMemcpy(d_queryValue, cond.queryValue, cond.columnInfo.count * cond.columnInfo.stride, cudaMemcpyHostToDevice);
            cond.queryValue = d_queryValue; // Replace host pointer with device pointer

            // Allocate device memory for the contiguous buffer
            size_t bufferSize = cond.columnInfo.count * cond.columnInfo.stride;
            char *d_buffer;
            cudaMalloc(&d_buffer, bufferSize);
            cudaMemcpy(d_buffer, cond.columnInfo.data, bufferSize, cudaMemcpyHostToDevice);

            // Free the host buffer
            delete[] static_cast<char *>(cond.columnInfo.data);

            cond.columnInfo.data = d_buffer;
            break;
        }
        else
        {

            cudaMalloc(&d_queryValue, valueSize);
            cudaMemcpy(d_queryValue, cond.queryValue, valueSize, cudaMemcpyHostToDevice);
            cond.queryValue = d_queryValue; // Replace host pointer with device pointer

            // Allocate and copy column data to GPU
            void *d_columnData;
            size_t dataSize = cond.columnInfo.count * valueSize;
            cudaMalloc(&d_columnData, dataSize);
            cudaMemcpy(d_columnData, cond.columnInfo.data, dataSize, cudaMemcpyHostToDevice);
            cond.columnInfo.data = d_columnData; // Replace host pointer with device pointer
        }
    }

    // Step 2: Copy conditions to GPU (now they contain device pointers)
    ConditionGPU *d_conditions;
    cudaMalloc(&d_conditions, conditions.size() * sizeof(ConditionGPU));
    cudaMemcpy(d_conditions, conditions.data(), conditions.size() * sizeof(ConditionGPU), cudaMemcpyHostToDevice);

    // Allocate memory for output flags
    bool *h_outputFlags = new bool[rowCount];
    std::fill(h_outputFlags, h_outputFlags + rowCount, false);

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
        d_outputFlags);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    for (auto &cond : conditions)
    {
        cudaFree(const_cast<void *>(cond.queryValue));
        cudaFree(cond.columnInfo.data);
    }
    cudaFree(d_conditions);

    cudaError_t err = cudaGetLastError(); // Check for errors
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return resultTable;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceSynchronize();

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

    delete[] h_outputFlags;

    cudaDeviceReset();

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

__device__ bool compare(GPUDBMS::ComparisonOperator op, bool a, bool b)
{
    switch (op)
    {
    case GPUDBMS::ComparisonOperator::EQUAL:
        return a == b;
    case GPUDBMS::ComparisonOperator::NOT_EQUAL:
        return a != b;
    case GPUDBMS::ComparisonOperator::LESS_THAN:
        return !a && b; // false < true
    case GPUDBMS::ComparisonOperator::GREATER_THAN:
        return a && !b; // true > false
    case GPUDBMS::ComparisonOperator::LESS_EQUAL:
        return !a || b; // a <= b is true if a is false or b is true
    case GPUDBMS::ComparisonOperator::GREATER_EQUAL:
        return a || !b; // a >= b is true if a is true or b is false
    default:
        return false;
    }
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

__device__ bool compareString(GPUDBMS::ComparisonOperator op, const char *a, const char *b)
{
    int cmp = device_strcmp(a, b);

    switch (op)
    {
    case GPUDBMS::ComparisonOperator::EQUAL:
        return cmp == 0;
    case GPUDBMS::ComparisonOperator::NOT_EQUAL:
        return cmp != 0;
    case GPUDBMS::ComparisonOperator::LESS_THAN:
        return cmp < 0;
    case GPUDBMS::ComparisonOperator::GREATER_THAN:
        return cmp > 0;
    case GPUDBMS::ComparisonOperator::LESS_EQUAL:
        return cmp <= 0;
    case GPUDBMS::ComparisonOperator::GREATER_EQUAL:
        return cmp >= 0;
    default:
        return false;
    }
}
