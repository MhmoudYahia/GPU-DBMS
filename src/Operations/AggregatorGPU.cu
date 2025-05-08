#include "../../include/Operations/AggregatorGPU.cuh"

// Atomic minimum for floats
__device__ static float atomicMinFloat(float *address, float val)
{
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// Atomic maximum for floats
__device__ static float atomicMaxFloat(float *address, float val)
{
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// Atomic minimum for doubles
__device__ static double atomicMinDouble(double *address, double val)
{
    unsigned long long *address_as_ull = (unsigned long long *)address;
    unsigned long long old = *address_as_ull, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Atomic maximum for doubles
__device__ static double atomicMaxDouble(double *address, double val)
{
    unsigned long long *address_as_ull = (unsigned long long *)address;
    unsigned long long old = *address_as_ull, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}



__global__ void aggregationKernel(
    int numRows,
    int numAggregations,
    const AggregationInfo *aggregations,
    const GroupByInfo *groupBy,
    bool *filterFlags)
{
    extern __shared__ char sharedMemory[];

    // Calculate required shared memory sizes
    size_t sharedMemPerAgg = 0;
    for (int aggIdx = 0; aggIdx < numAggregations; aggIdx++)
    {
        const AggregationInfo &agg = aggregations[aggIdx];
        switch (agg.dataType)
        {
        case GPUDBMS::DataType::INT:
            sharedMemPerAgg = (sharedMemPerAgg > sizeof(int)) ? sharedMemPerAgg : sizeof(int);
            break;
        case GPUDBMS::DataType::FLOAT:
            sharedMemPerAgg = (sharedMemPerAgg > sizeof(float)) ? sharedMemPerAgg : sizeof(float);
            break;
        case GPUDBMS::DataType::DOUBLE:
            sharedMemPerAgg = (sharedMemPerAgg > sizeof(double)) ? sharedMemPerAgg : sizeof(double);
            break;
        case GPUDBMS::DataType::STRING:
        case GPUDBMS::DataType::VARCHAR:
            sharedMemPerAgg = (sharedMemPerAgg > (size_t)256) ? sharedMemPerAgg : (size_t)256;
            break;
        case GPUDBMS::DataType::DATETIME:
            sharedMemPerAgg = (sharedMemPerAgg > (size_t)20) ? sharedMemPerAgg : (size_t)20;
            break;
        case GPUDBMS::DataType::DATE:
            sharedMemPerAgg = (sharedMemPerAgg > sizeof(int)) ? sharedMemPerAgg : sizeof(int);
            break;
        }
    }


    int groupId = groupBy ? blockIdx.x : 0;
    int numGroups = groupBy ? groupBy->numGroups : 1;

    if (groupId >= numGroups)
        return;

    int startRow = 0, endRow = numRows;
    if (groupBy)
    {
        startRow = groupId * numRows / numGroups;
        endRow = (groupId + 1) * numRows / numGroups;
    }

    // Initialize shared memory for aggregations
    for (int aggIdx = 0; aggIdx < numAggregations; aggIdx++)
    {
        const AggregationInfo &agg = aggregations[aggIdx];
        char *aggSharedMem = &sharedMemory[aggIdx * sharedMemPerAgg];

        switch (agg.type)
        {
        case AggregateFunction::SUM:
        case AggregateFunction::AVG:
            if (agg.dataType == GPUDBMS::DataType::INT)
            {
                *((int *)&sharedMemory[aggIdx * sizeof(int)]) = 0;
            }
            else if (agg.dataType == GPUDBMS::DataType::FLOAT)
            {
                *((float *)&sharedMemory[aggIdx * sizeof(float)]) = 0.0f;
            }
            else if (agg.dataType == GPUDBMS::DataType::DOUBLE)
            {
                *((double *)&sharedMemory[aggIdx * sizeof(double)]) = 0.0;
            }
            break;
        case AggregateFunction::MIN:
            if (agg.dataType == GPUDBMS::DataType::INT)
            {
                *((int *)&sharedMemory[aggIdx * sizeof(int)]) = INT_MAX;
            }
            else if (agg.dataType == GPUDBMS::DataType::FLOAT)
            {
                *((float *)&sharedMemory[aggIdx * sizeof(float)]) = FLT_MAX;
            }
            else if (agg.dataType == GPUDBMS::DataType::DOUBLE)
            {
                *((double *)&sharedMemory[aggIdx * sizeof(double)]) = DBL_MAX;
            }
            else if (agg.dataType == GPUDBMS::DataType::STRING ||
                     agg.dataType == GPUDBMS::DataType::VARCHAR)
            {
                // Initialize to a large string value
                char *str = (char *)&sharedMemory[aggIdx * sizeof(char) * 256]; // Assuming max string length of 256
                memset(str, 0, sizeof(char) * 256);
            }
            else if (agg.dataType == GPUDBMS::DataType::DATE)
            {
                // Initialize to a large date value
                *((int *)&sharedMemory[aggIdx * sizeof(int)]) = INT_MAX; // Assuming date is stored as an int
            }
            else if (agg.dataType == GPUDBMS::DataType::DATETIME)
            {
                // char *str = (char *)&sharedMemory[aggIdx * sizeof(char) * 20];
                // Initialize to "9999-12-31 23:59:59" (max datetime)
                // device_strcpy(str, "9999-12-31 23:59:59");
                device_strcpy(aggSharedMem, "9999-12-31 23:59:59");
            }
            break;
        case AggregateFunction::MAX:
            if (agg.dataType == GPUDBMS::DataType::INT)
            {
                *((int *)&sharedMemory[aggIdx * sizeof(int)]) = INT_MIN;
            }
            else if (agg.dataType == GPUDBMS::DataType::FLOAT)
            {
                *((float *)&sharedMemory[aggIdx * sizeof(float)]) = -FLT_MAX; // Use -FLT_MAX instead of FLT_MIN
            }
            else if (agg.dataType == GPUDBMS::DataType::DOUBLE)
            {
                *((double *)&sharedMemory[aggIdx * sizeof(double)]) = -DBL_MAX; // Use -DBL_MAX instead of DBL_MIN
            }
            else if (agg.dataType == GPUDBMS::DataType::STRING ||
                     agg.dataType == GPUDBMS::DataType::VARCHAR)
            {
                // Initialize to a small string value
                char *str = (char *)&sharedMemory[aggIdx * sizeof(char) * 256]; // Assuming max string length of 256
                memset(str, 0, sizeof(char) * 256);
            }
            else if (agg.dataType == GPUDBMS::DataType::DATE)
            {
                // Initialize to a small date value
                *((int *)&sharedMemory[aggIdx * sizeof(int)]) = INT_MIN; // Assuming date is stored as an int
            }
            else if (agg.dataType == GPUDBMS::DataType::DATETIME)
            {
                // char *str = (char *)&sharedMemory[aggIdx * sizeof(char) * 20];
                // Initialize to "0000-01-01 00:00:00" (min datetime)
                // device_strcpy(str, "0000-01-01 00:00:00");
                device_strcpy(aggSharedMem, "0000-01-01 00:00:00");
            }
            break;
        case AggregateFunction::COUNT:
            *((int *)&sharedMemory[aggIdx * sizeof(int)]) = 0;
            break;
        }
    }

    __syncthreads();

    for (int row = startRow + threadIdx.x; row < endRow; row += blockDim.x)
    {
        if (filterFlags && !filterFlags[row])
            continue;

        if (groupBy)
        {
            bool inGroup = false;
            switch (groupBy->keyType)
            {
            case GPUDBMS::DataType::INT:
            {
                inGroup = (groupBy->groupIndices[row] == groupId);
                break;
            }
                // Add cases for other data types
            }
            if (!inGroup)
                continue;
        }

        for (int aggIdx = 0; aggIdx < numAggregations; aggIdx++)
        {
            const AggregationInfo &agg = aggregations[aggIdx];
            char *aggSharedMem = &sharedMemory[aggIdx * sharedMemPerAgg];

            switch (agg.type)
            {
            case AggregateFunction::SUM:
            case AggregateFunction::AVG:
            {
                if (agg.dataType == GPUDBMS::DataType::INT)
                {
                    const int *data = static_cast<const int *>(agg.inputData);
                    atomicAdd((int *)&sharedMemory[aggIdx * sizeof(int)], data[row]);
                }
                else if (agg.dataType == GPUDBMS::DataType::FLOAT)
                {
                    const float *data = static_cast<const float *>(agg.inputData);
                    atomicAdd((float *)&sharedMemory[aggIdx * sizeof(float)], data[row]);
                }
                else if (agg.dataType == GPUDBMS::DataType::DOUBLE)
                {
                    const double *data = static_cast<const double *>(agg.inputData);
                    atomicAdd((double *)&sharedMemory[aggIdx * sizeof(double)], data[row]);
                }
                break;
            }
            case AggregateFunction::MIN:
            {
                if (agg.dataType == GPUDBMS::DataType::INT)
                {
                    const int *data = static_cast<const int *>(agg.inputData);
                    atomicMin((int *)&sharedMemory[aggIdx * sizeof(int)], data[row]);
                }
                else if (agg.dataType == GPUDBMS::DataType::FLOAT)
                {
                    const float *data = static_cast<const float *>(agg.inputData);
                    atomicMinFloat((float *)&sharedMemory[aggIdx * sizeof(float)], data[row]);
                }
                else if (agg.dataType == GPUDBMS::DataType::DOUBLE)
                {
                    const double *data = static_cast<const double *>(agg.inputData);
                    atomicMinDouble((double *)&sharedMemory[aggIdx * sizeof(double)], data[row]);
                }
                else if (agg.dataType == GPUDBMS::DataType::STRING ||
                         agg.dataType == GPUDBMS::DataType::VARCHAR)
                {
                    // Handle string min
                    char *data = (char *)agg.inputData;
                    char *minStr = (char *)&sharedMemory[aggIdx * sizeof(char) * 256]; // Assuming max string length of 256
                    if (device_strcmp(data + row * 256, minStr) < 0)
                    {
                        device_strcpy(minStr, data + row * 256);
                    }
                }
                else if (agg.dataType == GPUDBMS::DataType::DATE)
                {
                    // Handle date min
                    const int *data = static_cast<const int *>(agg.inputData);
                    atomicMin((int *)&sharedMemory[aggIdx * sizeof(int)], data[row]);
                }
                else if (agg.dataType == GPUDBMS::DataType::DATETIME)
                {
                    const char *rowData = ((const char *)agg.inputData) + row * 20; // Use actual stride

                    // char *data = (char *)agg.inputData;
                    // char *minStr = (char *)&sharedMemory[aggIdx * sizeof(char) * 20];
                    
                    if(device_datetime_cmp(rowData, aggSharedMem) < 0)
                    {
                        device_strcpy(aggSharedMem, rowData);
                    }
                }
                break;
            }
            case AggregateFunction::MAX:
            {
                if (agg.dataType == GPUDBMS::DataType::INT)
                {
                    const int *data = static_cast<const int *>(agg.inputData);
                    atomicMax((int *)&sharedMemory[aggIdx * sizeof(int)], data[row]);
                }
                else if (agg.dataType == GPUDBMS::DataType::FLOAT)
                {
                    const float *data = static_cast<const float *>(agg.inputData);
                    atomicMaxFloat((float *)&sharedMemory[aggIdx * sizeof(float)], data[row]);
                }
                else if (agg.dataType == GPUDBMS::DataType::DOUBLE)
                {
                    const double *data = static_cast<const double *>(agg.inputData);
                    atomicMaxDouble((double *)&sharedMemory[aggIdx * sizeof(double)], data[row]);
                }
                else if (agg.dataType == GPUDBMS::DataType::STRING ||
                         agg.dataType == GPUDBMS::DataType::VARCHAR)
                {
                    // Handle string max
                    char *data = (char *)agg.inputData;
                    char *maxStr = (char *)&sharedMemory[aggIdx * sizeof(char) * 256]; // Assuming max string length of 256
                    if (device_strcmp(data + row * 256, maxStr) > 0)
                    {
                        device_strcpy(maxStr, data + row * 256);
                    }
                }
                else if (agg.dataType == GPUDBMS::DataType::DATE)
                {
                    // Handle date max
                    const int *data = static_cast<const int *>(agg.inputData);
                    atomicMax((int *)&sharedMemory[aggIdx * sizeof(int)], data[row]);
                }
                else if (agg.dataType == GPUDBMS::DataType::DATETIME)
                {
                    // char *data = (char *)agg.inputData;
                    // char *maxStr = (char *)&sharedMemory[aggIdx * sizeof(char) * 20];
                    // if (device_datetime_cmp(data + row * 20, maxStr) > 0)
                    // {
                    //     device_strcpy(maxStr, data + row * 20);
                    // }

                    const char *rowData = ((const char *)agg.inputData) + row * 20; // Use actual stride

                    if(device_datetime_cmp(rowData, aggSharedMem) > 0)
                    {
                        device_strcpy(aggSharedMem, rowData);
                    }
                }
                break;
            }
            case AggregateFunction::COUNT:
            {
                atomicAdd((int *)&sharedMemory[aggIdx * sizeof(int)], 1);

                break;
            }
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        for (int aggIdx = 0; aggIdx < numAggregations; aggIdx++)
        {
            const AggregationInfo &agg = aggregations[aggIdx];

            switch (agg.type)
            {
            case AggregateFunction::SUM:
            case AggregateFunction::COUNT:
            {
                if (agg.dataType == GPUDBMS::DataType::INT)
                {
                    ((int *)agg.outputData)[groupId] = *((int *)&sharedMemory[aggIdx * sizeof(int)]);
                }
                else if (agg.dataType == GPUDBMS::DataType::FLOAT)
                {
                    ((float *)agg.outputData)[groupId] = *((float *)&sharedMemory[aggIdx * sizeof(float)]);
                }
                else if (agg.dataType == GPUDBMS::DataType::DOUBLE)
                {
                    ((double *)agg.outputData)[groupId] = *((double *)&sharedMemory[aggIdx * sizeof(double)]);
                }
                else if (agg.dataType == GPUDBMS::DataType::STRING ||
                         agg.dataType == GPUDBMS::DataType::VARCHAR)
                {
                    // Handle string output
                    char *str = (char *)agg.outputData;
                    char *resultStr = (char *)&sharedMemory[aggIdx * sizeof(char) * 256]; // Assuming max string length of 256
                    device_strcpy(str + groupId * 256, resultStr);
                }
                else if (agg.dataType == GPUDBMS::DataType::DATE)
                {
                    ((int *)agg.outputData)[groupId] = *((int *)&sharedMemory[aggIdx * sizeof(int)]);
                }
                else if (agg.dataType == GPUDBMS::DataType::DATETIME)
                {
                    // Handle datetime output
                    char *str = (char *)agg.outputData;
                    char *resultStr = (char *)&sharedMemory[aggIdx * sizeof(char) * 20]; // Assuming max datetime string length of 20
                    device_strcpy(str + groupId * 20, resultStr);
                }
                break;
            }
            case AggregateFunction::AVG:
            {
                int countIdx = -1;
                // Find the COUNT aggregation (simplified - in real code you'd want a better way)
                for (int i = 0; i < numAggregations; i++)
                {
                    if (aggregations[i].type == AggregateFunction::COUNT)
                    {
                        countIdx = i;
                        break;
                    }
                }
                int count = countIdx >= 0 ? *((int *)&sharedMemory[countIdx * sizeof(int)]) : (endRow - startRow);

                if (count > 0)
                {
                    if (agg.dataType == GPUDBMS::DataType::INT)
                    {
                        int sum = *((int *)&sharedMemory[aggIdx * sizeof(int)]);
                        ((int *)agg.outputData)[groupId] = sum / count;
                    }
                    else if (agg.dataType == GPUDBMS::DataType::FLOAT)
                    {
                        float sum = *((float *)&sharedMemory[aggIdx * sizeof(float)]);
                        ((float *)agg.outputData)[groupId] = sum / count;
                    }
                    else if (agg.dataType == GPUDBMS::DataType::DOUBLE)
                    {
                        double sum = *((double *)&sharedMemory[aggIdx * sizeof(double)]);
                        ((double *)agg.outputData)[groupId] = sum / count;
                    }
                }
                break;
            }
            case AggregateFunction::MIN:
            case AggregateFunction::MAX:
            {
                if (agg.dataType == GPUDBMS::DataType::INT)
                {
                    ((int *)agg.outputData)[groupId] = *((int *)&sharedMemory[aggIdx * sizeof(int)]);
                }
                else if (agg.dataType == GPUDBMS::DataType::FLOAT)
                {
                    ((float *)agg.outputData)[groupId] = *((float *)&sharedMemory[aggIdx * sizeof(float)]);
                }
                else if (agg.dataType == GPUDBMS::DataType::DOUBLE)
                {
                    ((double *)agg.outputData)[groupId] = *((double *)&sharedMemory[aggIdx * sizeof(double)]);
                }
                else if (agg.dataType == GPUDBMS::DataType::STRING ||
                         agg.dataType == GPUDBMS::DataType::VARCHAR)
                {
                    // Handle string output
                    char *str = (char *)agg.outputData;
                    char *resultStr = (char *)&sharedMemory[aggIdx * sizeof(char) * 256]; // Assuming max string length of 256
                    device_strcpy(str + groupId * 256, resultStr);
                }
                else if (agg.dataType == GPUDBMS::DataType::DATE)
                {
                    ((int *)agg.outputData)[groupId] = *((int *)&sharedMemory[aggIdx * sizeof(int)]);
                }
                else if (agg.dataType == GPUDBMS::DataType::DATETIME)
                {
                    // Handle datetime output
                    char *str = (char *)agg.outputData;
                    char *resultStr = (char *)&sharedMemory[aggIdx * sizeof(char) * 20]; // Assuming max datetime string length of 20
                    device_strcpy(str + groupId * 20, resultStr);
                }
                break;
            }
            }
        }
    }
}

extern "C" GPUDBMS::Table launchAggregationKernel(
    const GPUDBMS::Table &inputTable,
    const std::vector<std::string> &groupByColumns,
    const std::vector<std::pair<std::string, AggregateFunction>> &aggregations,
    const std::vector<GPUDBMS::Condition> &conditions)
{
    const size_t rowCount = inputTable.getRowCount();
    const size_t numAggregations = aggregations.size();
    const size_t numGroupByColumns = groupByColumns.size();

    // Validate inputs
    if (rowCount == 0 || numAggregations == 0)
    {
        throw std::runtime_error("Invalid input: empty table or no aggregations specified");
    }

    // Create column name to index mapping
    std::unordered_map<std::string, int> columnNameToIndex;
    const auto &columns = inputTable.getColumns();
    for (size_t i = 0; i < columns.size(); ++i)
    {
        columnNameToIndex[columns[i].getName()] = static_cast<int>(i);
    }

    // Prepare filter flags if conditions are provided
    bool *d_filterFlags = nullptr;
    if (!conditions.empty())
    {
        // TODO: Implement condition evaluation similar to selectKernel
        // For now, we'll just allocate device memory for filter flags
        cudaMalloc(&d_filterFlags, rowCount * sizeof(bool));
        // Initialize all to true (include all rows)
        cudaMemset(d_filterFlags, true, rowCount * sizeof(bool));
    }

    // Prepare group by information
    GroupByInfo groupByInfo = {
        .keyType = GPUDBMS::DataType::INT, // Default type, will be set based on group by columns
        .keyData = nullptr,
        .numGroups = 1,
        .groupIndices = nullptr,
    };
    std::vector<GPUDBMS::ColumnInfoGPU> groupByColumnInfos;
    std::vector<int> h_groupIndices(rowCount, 0); // Default to single group if no group by

    if (!groupByColumns.empty())
    {
        // Get group by column info and copy data to device
        for (const auto &colName : groupByColumns)
        {
            auto it = columnNameToIndex.find(colName);
            if (it == columnNameToIndex.end())
            {
                throw std::runtime_error("Group by column not found: " + colName);
            }

            GPUDBMS::ColumnInfoGPU colInfo = inputTable.getColumnInfoGPU(colName);

            // Allocate and copy column data to GPU
            void *d_columnData = nullptr;
            size_t dataSize = colInfo.count * colInfo.stride;
            cudaMalloc(&d_columnData, dataSize);

            // Special handling for string and datetime types
            if (colInfo.type == GPUDBMS::DataType::DATETIME ||
                colInfo.type == GPUDBMS::DataType::DATE ||
                colInfo.type == GPUDBMS::DataType::STRING ||
                colInfo.type == GPUDBMS::DataType::VARCHAR)
            {
                size_t elementSize = (colInfo.type == GPUDBMS::DataType::DATETIME) ? 20 : 256;
                size_t totalSize = rowCount * elementSize;
                cudaMalloc(&d_columnData, totalSize);
                cudaMemcpy(d_columnData, colInfo.data, totalSize, cudaMemcpyHostToDevice);
                colInfo.data = d_columnData;
            }
            else
            {
                cudaMemcpy(d_columnData, colInfo.data, dataSize, cudaMemcpyHostToDevice);
                colInfo.data = d_columnData;
            }

            groupByColumnInfos.push_back(colInfo);
        }

        // For simplicity, we'll assume single grouping column for now
        // Determine groups based on unique values in the group by column
        std::unordered_set<int> uniqueGroups;
        const int *groupByData = static_cast<const int *>(groupByColumnInfos[0].data);

        // Copy data back to host temporarily to determine groups
        std::vector<int> h_groupByData(rowCount);
        cudaMemcpy(h_groupByData.data(), groupByColumnInfos[0].data,
                   rowCount * sizeof(int), cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < rowCount; ++i)
        {
            uniqueGroups.insert(h_groupByData[i]);
        }

        int numGroups = static_cast<int>(uniqueGroups.size());

        // Map each row to its group index
        std::unordered_map<int, int> groupMapping;
        int groupIndex = 0;
        for (const int groupKey : uniqueGroups)
        {
            groupMapping[groupKey] = groupIndex++;
        }

        for (size_t i = 0; i < rowCount; ++i)
        {
            h_groupIndices[i] = groupMapping[h_groupByData[i]];
        }

        groupByInfo.keyType = groupByColumnInfos[0].type;
        groupByInfo.numGroups = numGroups;

        // Allocate and copy group indices to device
        int *d_groupIndices = nullptr;
        cudaMalloc(&d_groupIndices, rowCount * sizeof(int));
        cudaMemcpy(d_groupIndices, h_groupIndices.data(), rowCount * sizeof(int), cudaMemcpyHostToDevice);

        groupByInfo.groupIndices = d_groupIndices;
        groupByInfo.keyData = groupByColumnInfos[0].data; // Use the already copied device data
    }

    // Prepare aggregation information
    std::vector<AggregationInfo> aggregationInfos;
    std::vector<GPUDBMS::ColumnInfoGPU> aggColumnInfos;

    for (const auto &agg : aggregations)
    {
        const std::string &colName = agg.first;
        auto it = columnNameToIndex.find(colName);
        if (it == columnNameToIndex.end())
        {
            throw std::runtime_error("Aggregation column not found: " + colName);
        }

        GPUDBMS::ColumnInfoGPU colInfo = inputTable.getColumnInfoGPU(colName);

        // Allocate and copy input data to device
        void *d_inputData = nullptr;
        size_t dataSize = colInfo.count * colInfo.stride;
        cudaMalloc(&d_inputData, dataSize);

        if (colInfo.type == GPUDBMS::DataType::DATETIME ||
            colInfo.type == GPUDBMS::DataType::DATE ||
            colInfo.type == GPUDBMS::DataType::STRING ||
            colInfo.type == GPUDBMS::DataType::VARCHAR)
        {
            size_t elementSize = (colInfo.type == GPUDBMS::DataType::DATETIME) ? 20 : 256;
            size_t totalSize = rowCount * elementSize;
            cudaMalloc(&d_inputData, totalSize);
            cudaMemcpy(d_inputData, colInfo.data, totalSize, cudaMemcpyHostToDevice);
        }
        else
        {
            size_t elementSize = getTypeSize(colInfo.type);
            cudaMemcpy(d_inputData, colInfo.data, rowCount * elementSize, cudaMemcpyHostToDevice);
        }
       
        
        // Update column info with device pointer
        colInfo.data = d_inputData;
        aggColumnInfos.push_back(colInfo);

        AggregationInfo aggInfo;
        aggInfo.type = agg.second;
        aggInfo.dataType = colInfo.type;
        aggInfo.inputData = d_inputData;

        // Allocate output memory on device
        size_t outputSize = groupByInfo.numGroups * getTypeSize(colInfo.type);
        void *d_outputData = nullptr;
        cudaMalloc(&d_outputData, outputSize);

        // Initialize output memory based on aggregation type
        if (aggInfo.type == AggregateFunction::MIN && aggInfo.dataType == GPUDBMS::DataType::DATETIME)
        {
            std::vector<std::string> init(groupByInfo.numGroups, "9999-12-31 23:59:59");
            cudaMemcpy(d_outputData, init.data(), outputSize, cudaMemcpyHostToDevice);
        }
        else if (aggInfo.type == AggregateFunction::MAX && aggInfo.dataType == GPUDBMS::DataType::DATETIME)
        {
            std::vector<std::string> init(groupByInfo.numGroups, "0000-01-01 00:00:00");
            cudaMemcpy(d_outputData, init.data(), outputSize, cudaMemcpyHostToDevice);
        }
        else
        {
            cudaMemset(d_outputData, 0, outputSize);
        }

        aggInfo.outputData = d_outputData;
        aggInfo.outputSize = outputSize;

        aggregationInfos.push_back(aggInfo);
    }
    // Copy aggregation info to device
    AggregationInfo *d_aggregations = nullptr;
    cudaMalloc(&d_aggregations, numAggregations * sizeof(AggregationInfo));
    cudaMemcpy(d_aggregations, aggregationInfos.data(),
               numAggregations * sizeof(AggregationInfo), cudaMemcpyHostToDevice);

    // Copy group by info to device if needed
    GroupByInfo *d_groupBy = nullptr;
    if (!groupByColumns.empty())
    {
        cudaMalloc(&d_groupBy, sizeof(GroupByInfo));
        cudaMemcpy(d_groupBy, &groupByInfo, sizeof(GroupByInfo), cudaMemcpyHostToDevice);
    }

    // Calculate shared memory size needed
    size_t sharedMemSize = numAggregations * sizeof(double); // Worst case size

    // Launch kernel
    int blockSize = 256;
    int numGroups = groupByInfo.numGroups;
    int numBlocks = (numGroups + blockSize - 1) / blockSize; // Adjust blocks based on group count

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    aggregationKernel<<<numBlocks, blockSize, sharedMemSize>>>(
        rowCount,
        numAggregations,
        d_aggregations,
        d_groupBy,
        d_filterFlags);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Aggregation execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error in aggregation kernel: " << cudaGetErrorString(err) << std::endl;
        // Clean up and return empty table
        GPUDBMS::Table emptyTable;
        return emptyTable;
    }

    // Create output table with aggregation results
    GPUDBMS::Table resultTable;

    // Add group by columns first if they exist
    if (!groupByColumns.empty())
    {
        // Copy group keys back to host
        std::vector<int> h_groupKeys(groupByInfo.numGroups);
        cudaMemcpy(h_groupKeys.data(), groupByColumnInfos[0].data,
                   groupByInfo.numGroups * sizeof(int), cudaMemcpyDeviceToHost);

        // Add group by column to result table
        resultTable.addColumn(GPUDBMS::Column(groupByColumns[0], groupByColumnInfos[0].type));

        // Add group keys to the result table
        for (int key : h_groupKeys)
        {
            resultTable.appendIntValue(0, key);
        }
    }

    // Add aggregation result columns and copy results
    for (size_t i = 0; i < aggregations.size(); i++)
    {
        const auto &agg = aggregations[i];
        const AggregationInfo &aggInfo = aggregationInfos[i];

        // Create column name based on aggregation type
        std::string colName = agg.first + "_";
        switch (agg.second)
        {
        case AggregateFunction::SUM:
            colName += "sum";
            break;
        case AggregateFunction::AVG:
            colName += "avg";
            break;
        case AggregateFunction::MIN:
            colName += "min";
            break;
        case AggregateFunction::MAX:
            colName += "max";
            break;
        case AggregateFunction::COUNT:
            colName += "count";
            break;
        }

        // Add column to result table
        resultTable.addColumn(GPUDBMS::Column(colName, aggInfo.dataType));

        // Copy results back to host based on data type
        switch (aggInfo.dataType)
        {
        case GPUDBMS::DataType::INT:
        {
            std::vector<int> h_results(groupByInfo.numGroups);
            cudaMemcpy(h_results.data(), aggInfo.outputData,
                       groupByInfo.numGroups * sizeof(int), cudaMemcpyDeviceToHost);
            for (int val : h_results)
            {
                resultTable.appendIntValue(i + groupByColumns.size(), val);
            }
            break;
        }
        case GPUDBMS::DataType::FLOAT:
        {
            std::vector<float> h_results(groupByInfo.numGroups);
            cudaMemcpy(h_results.data(), aggInfo.outputData,
                       groupByInfo.numGroups * sizeof(float), cudaMemcpyDeviceToHost);
            for (float val : h_results)
            {
                resultTable.appendFloatValue(i + groupByColumns.size(), val);
            }
            break;
        }
        case GPUDBMS::DataType::DOUBLE:
        {
            std::vector<double> h_results(groupByInfo.numGroups);
            cudaMemcpy(h_results.data(), aggInfo.outputData,
                       groupByInfo.numGroups * sizeof(double), cudaMemcpyDeviceToHost);
            for (double val : h_results)
            {
                resultTable.appendDoubleValue(i + groupByColumns.size(), val);
            }
            break;
        }
        case GPUDBMS::DataType::STRING:
        case GPUDBMS::DataType::VARCHAR:
        {
            std::vector<char> h_results(groupByInfo.numGroups * 256); // Assuming max string length of 256
            cudaMemcpy(h_results.data(), aggInfo.outputData,
                       groupByInfo.numGroups * 256, cudaMemcpyDeviceToHost);
            for (int j = 0; j < groupByInfo.numGroups; ++j)
            {
                resultTable.appendStringValue(i + groupByColumns.size(),
                                              std::string(h_results.data() + j * 256, 256));
            }
            break;
        }
        case GPUDBMS::DataType::DATE:
        {
            std::vector<int> h_results(groupByInfo.numGroups);
            cudaMemcpy(h_results.data(), aggInfo.outputData,
                       groupByInfo.numGroups * sizeof(int), cudaMemcpyDeviceToHost);
            for (int val : h_results)
            {
                resultTable.appendIntValue(i + groupByColumns.size(), val);
            }
            break;
        }
        case GPUDBMS::DataType::DATETIME:
        {
            size_t elementSize = 20; // or use columnInfo.stride if it's correct
            size_t totalSize = groupByInfo.numGroups * elementSize;
            std::vector<char> h_results(totalSize);
            cudaMemcpy(h_results.data(), aggInfo.outputData, totalSize, cudaMemcpyDeviceToHost);

            for (int j = 0; j < groupByInfo.numGroups; ++j)
            {
                const char *datetimeStr = h_results.data() + j * elementSize;
                size_t len = strnlen(datetimeStr, elementSize);
                resultTable.appendStringValue(i + groupByColumns.size(),
                                              std::string(datetimeStr, len));
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type for aggregation result");
        }
    }

    // Free device memory
    for (auto &aggInfo : aggregationInfos)
    {
        cudaFree(aggInfo.inputData);
        cudaFree(aggInfo.outputData);
    }
    cudaFree(d_aggregations);

    if (d_groupBy)
    {
        cudaFree(d_groupBy);
    }

    for (auto &colInfo : groupByColumnInfos)
    {
        cudaFree(const_cast<void *>(colInfo.data));
    }

    if (d_filterFlags)
    {
        cudaFree(d_filterFlags);
    }

    return resultTable;
}