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
        // // First evaluate conditions on CPU (or could use your selectKernel)
        // std::vector<bool> h_filterFlags(rowCount, true);
        // // ... evaluate conditions to fill h_filterFlags ...

        // cudaMalloc(&d_filterFlags, rowCount * sizeof(bool));
        // cudaMemcpy(d_filterFlags, h_filterFlags.data(), rowCount * sizeof(bool), cudaMemcpyHostToDevice);
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
        // Get group by column info
        for (const auto &colName : groupByColumns)
        {
            auto it = columnNameToIndex.find(colName);
            if (it == columnNameToIndex.end())
            {
                throw std::runtime_error("Group by column not found: " + colName);
            }
            groupByColumnInfos.push_back(inputTable.getColumnInfoGPU(colName));
        }

        // Determine groups based on unique values in the group by column
        std::unordered_set<int> uniqueGroups;
        const int *groupByData = static_cast<const int *>(groupByColumnInfos[0].data);

        for (size_t i = 0; i < rowCount; ++i)
        {
            uniqueGroups.insert(groupByData[i]);
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
            h_groupIndices[i] = groupMapping[groupByData[i]];
        }

        groupByInfo.keyType = groupByColumnInfos[0].type; // Assuming single grouping column
        groupByInfo.numGroups = numGroups;

        // Allocate and copy group indices to device
        int *d_groupIndices = nullptr;
        cudaMalloc(&d_groupIndices, rowCount * sizeof(int));
        cudaMemcpy(d_groupIndices, h_groupIndices.data(), rowCount * sizeof(int), cudaMemcpyHostToDevice);

        groupByInfo.groupIndices = d_groupIndices;

        // Allocate and copy group by column data to device
        void *d_groupByData = nullptr;
        size_t dataSize = getTypeSize(groupByColumnInfos[0].type) * rowCount;
        cudaMalloc(&d_groupByData, dataSize);
        cudaMemcpy(d_groupByData, groupByColumnInfos[0].data, dataSize, cudaMemcpyHostToDevice);
        groupByInfo.keyData = d_groupByData;
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
        aggColumnInfos.push_back(colInfo);

        AggregationInfo aggInfo;
        aggInfo.type = agg.second;
        aggInfo.dataType = colInfo.type;

        // Allocate and copy input data to device
        void *d_inputData = nullptr;
        size_t dataSize = getTypeSize(colInfo.type) * rowCount;
        cudaMalloc(&d_inputData, dataSize);
        cudaMemcpy(d_inputData, colInfo.data, dataSize, cudaMemcpyHostToDevice);
        aggInfo.inputData = d_inputData;

        // Allocate output memory
        size_t outputSize = groupByInfo.numGroups * getTypeSize(colInfo.type);
        void *d_outputData = nullptr;
        cudaMalloc(&d_outputData, outputSize);
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
    int numBlocks = numGroups; // One block per group

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

    // Create output table with aggregation results
    GPUDBMS::Table resultTable;

    // Add group by columns first if they exist
    if (!groupByColumns.empty())
    {
        // For each group by column, we need to get the representative value per group
        // This is simplified - in reality you'd need to collect the group keys
        for (size_t i = 0; i < groupByColumns.size(); i++)
        {
            std::vector<int> groupKeys(groupByInfo.numGroups);
            // ... populate groupKeys with representative values ...

            resultTable.addColumn(GPUDBMS::Column(groupByColumnInfos[i].name, groupByColumnInfos[i].type));
        }
    }

    // Add aggregation result columns
    for (size_t i = 0; i < aggregations.size(); i++)
    {
        const auto &agg = aggregations[i];
        const AggregationInfo &aggInfo = aggregationInfos[i];

        // Copy results back to host
        std::vector<float> h_results(groupByInfo.numGroups);
        cudaMemcpy(h_results.data(), aggInfo.outputData,
                   aggInfo.outputSize, cudaMemcpyDeviceToHost);

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

        resultTable.addColumn(GPUDBMS::Column(colName, aggInfo.dataType));
    }

    // Copy results to output table
    for (size_t i = 0; i < resultTable.getColumnCount(); i++)
    {
        const auto &col = resultTable.getColumns()[i];
        if (col.getName().find("_") != std::string::npos)
        {
            // This is an aggregation result column
            size_t groupSize = groupByInfo.numGroups * getTypeSize(col.getType());
            void *h_data = malloc(groupSize);
            cudaMemcpy(h_data, aggregationInfos[i].outputData, groupSize, cudaMemcpyDeviceToHost);

            for (size_t j = 0; j < groupByInfo.numGroups; j++)
            {
                switch (col.getType())
                {
                case GPUDBMS::DataType::INT:
                    resultTable.appendIntValue(i, static_cast<int *>(h_data)[j]);
                    break;
                case GPUDBMS::DataType::FLOAT:
                    resultTable.appendFloatValue(i, static_cast<float *>(h_data)[j]);
                    break;
                case GPUDBMS::DataType::DOUBLE:
                    resultTable.appendDoubleValue(i, static_cast<double *>(h_data)[j]);
                    break;
                }
            }
            

        }
        else
        {
            // This is a group by column
            // ... handle group by column data ...
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
        GroupByInfo h_groupBy;
        cudaMemcpy(&h_groupBy, d_groupBy, sizeof(GroupByInfo), cudaMemcpyDeviceToHost);

        if (h_groupBy.keyData)
        {
            cudaFree(const_cast<void *>(h_groupBy.keyData));
        }
        if (h_groupBy.groupIndices)
        {
            cudaFree(h_groupBy.groupIndices);
        }
        cudaFree(d_groupBy);
    }

    if (d_filterFlags)
    {
        cudaFree(d_filterFlags);
    }

    return resultTable;
}