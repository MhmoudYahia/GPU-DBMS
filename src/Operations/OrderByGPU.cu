#include "../../include/Operations/OrderByGPU.cuh"

__global__ void generateSortKeysKernel(
    int numRows,
    const GPUDBMS::ColumnInfoGPU *sortColumns,
    int numSortColumns,
    SortOrder *sortDirections, // true=asc, false=desc
    uint64_t *keys)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows)
        return;

    uint64_t compositeKey = 0;

    for (int i = 0; i < numSortColumns; i++)
    {
        const GPUDBMS::ColumnInfoGPU &col = sortColumns[i];
        SortOrder order = sortDirections[i];

        switch (col.type)
        {
        case GPUDBMS::DataType::INT:
        {
            const int *data = static_cast<const int *>(col.data);
            uint32_t val = static_cast<uint32_t>(data[row]);
            if (order == SortOrder::DESC)
                val = ~val; // For descending order
            compositeKey = (compositeKey << 32) | val;
            break;
        }
        case GPUDBMS::DataType::FLOAT:
        {
            const float *data = static_cast<const float *>(col.data);
            uint32_t val = __float_as_uint(data[row]);
            // Flip bits if negative for proper float comparison
            val ^= (-(int)(val >> 31) | 0x80000000);
            if (order == SortOrder::DESC)
                val = ~val;
            compositeKey = (compositeKey << 32) | val;
            break;
        }
            // Similar cases for other data types
        }
    }

    keys[row] = compositeKey;
}

void sortRows(
    uint64_t *keys,
    uint32_t *indices,
    int numRows,
    cudaStream_t stream)
{
    // Use CUB or Thrust to sort keys and produce indices
    // Example with Thrust:
    thrust::device_ptr<uint64_t> keys_ptr(keys);
    thrust::device_ptr<uint32_t> indices_ptr(indices);
    thrust::sequence(indices_ptr, indices_ptr + numRows);
    thrust::sort_by_key(thrust::cuda::par.on(stream),
                        keys_ptr, keys_ptr + numRows,
                        indices_ptr);
}

__global__ void reorderDataKernel(
    int numRows,
    const uint32_t *indices,
    const GPUDBMS::ColumnInfoGPU *columnsToReorder,
    int numColumns,
    void **reorderedData)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows)
        return;

    uint32_t src_row = indices[row];

    for (int i = 0; i < numColumns; i++)
    {
        const GPUDBMS::ColumnInfoGPU &col = columnsToReorder[i];
        char *dst = static_cast<char *>(reorderedData[i]);
        const char *src = static_cast<const char *>(col.data);

        switch (col.type)
        {
        case GPUDBMS::DataType::INT:
            reinterpret_cast<int *>(dst)[row] =
                reinterpret_cast<const int *>(src)[src_row];
            break;
        }
    }
}

extern "C" GPUDBMS::Table launchOrderByKernel(
    const GPUDBMS::Table &inputTable,
    const std::vector<std::string> &sortColumns,
    const std::vector<SortOrder> &sortDirections)
{
    const size_t rowCount = inputTable.getRowCount();
    const size_t numSortColumns = sortColumns.size();

    // Validate inputs
    if (rowCount == 0 || numSortColumns == 0 || numSortColumns != sortDirections.size())
    {
        return inputTable; // Return original table if invalid input
    }

    // Create column name to index mapping
    std::unordered_map<std::string, int> columnNameToIndex;
    const auto &columns = inputTable.getColumns();
    for (size_t i = 0; i < columns.size(); ++i)
    {
        columnNameToIndex[columns[i].getName()] = static_cast<int>(i);
    }

    // Prepare sort column information
    std::vector<GPUDBMS::ColumnInfoGPU> sortColumnInfos;
    for (const auto &colName : sortColumns)
    {
        auto it = columnNameToIndex.find(colName);
        if (it == columnNameToIndex.end())
        {
            throw std::runtime_error("Column not found: " + colName);
        }
        sortColumnInfos.push_back(inputTable.getColumnInfoGPU(colName));
    }

    // Allocate device memory for keys and indices
    uint64_t *d_keys = nullptr;
    uint32_t *d_indices = nullptr;
    cudaMalloc((void **)&d_keys, rowCount * sizeof(uint64_t));
    cudaMalloc((void **)&d_indices, rowCount * sizeof(uint32_t));

    // Allocate device memory for sort columns and directions
    GPUDBMS::ColumnInfoGPU *d_sortColumns = nullptr;
    SortOrder *d_sortDirections = nullptr;
    cudaMalloc((void **)&d_sortColumns, numSortColumns * sizeof(GPUDBMS::ColumnInfoGPU));
    cudaMalloc((void **)&d_sortDirections, numSortColumns * sizeof(SortOrder));

    // Create a copy of column infos that we'll modify to point to device memory
    std::vector<GPUDBMS::ColumnInfoGPU> device_sortColumns = sortColumnInfos;
    for (size_t i = 0; i < numSortColumns; i++)
    {
        // Allocate and copy column data
        void *d_columnData = nullptr;
        size_t dataSize = rowCount * getTypeSize(sortColumnInfos[i].type);

        if (sortColumnInfos[i].type == GPUDBMS::DataType::STRING ||
            sortColumnInfos[i].type == GPUDBMS::DataType::VARCHAR ||
            sortColumnInfos[i].type == GPUDBMS::DataType::DATE ||
            sortColumnInfos[i].type == GPUDBMS::DataType::DATETIME)
        {
            // For variable-length types, use count * stride
            dataSize = sortColumnInfos[i].count * sortColumnInfos[i].stride;
            cudaMalloc(&d_columnData, dataSize);
            cudaMemcpy(d_columnData, sortColumnInfos[i].data, dataSize, cudaMemcpyHostToDevice);
        }
        else
        {
            // For fixed-length types
            cudaMalloc(&d_columnData, dataSize);
            cudaMemcpy(d_columnData, sortColumnInfos[i].data, dataSize, cudaMemcpyHostToDevice);
        }
        device_sortColumns[i].data = d_columnData;
    }

    // Copy data to device
    cudaMemcpy(d_sortColumns, device_sortColumns.data(),
               numSortColumns * sizeof(GPUDBMS::ColumnInfoGPU), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sortDirections, sortDirections.data(),
               numSortColumns * sizeof(SortOrder), cudaMemcpyHostToDevice);

    // Launch key generation kernel
    int blockSize = 256;
    int numBlocks = (rowCount + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    generateSortKeysKernel<<<numBlocks, blockSize>>>(
        rowCount,
        d_sortColumns,
        numSortColumns,
        d_sortDirections,
        d_keys);

    // Sort the keys and produce indices
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    sortRows(d_keys, d_indices, rowCount, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Sort execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy indices back to host
    std::vector<uint32_t> h_indices(rowCount);
    cudaMemcpy(h_indices.data(), d_indices, rowCount * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Convert to vector<int> for getSlicedTable
    std::vector<int> sortedIndices(h_indices.begin(), h_indices.end());

    // Free device memory
    for (auto &col : device_sortColumns)
    {
        cudaFree(const_cast<void *>(col.data));
    }
    cudaFree(d_sortColumns);
    cudaFree(d_sortDirections);
    cudaFree(d_keys);
    cudaFree(d_indices);

    // Return sorted table
    return inputTable.getSlicedTable(sortedIndices);
}