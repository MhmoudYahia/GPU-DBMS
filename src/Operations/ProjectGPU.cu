#include "../../include/Operations/ProjectGPU.cuh"

__global__ void projectKernel(
    int numRows,
    const GPUDBMS::ColumnInfoGPU *inputColumns,
    int numInputColumns,
    const int *projectionIndices, // indices of columns to project
    int numProjectionColumns,
    GPUDBMS::ColumnInfoGPU *outputColumns)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= numRows)
        return;

    for (int i = 0; i < numProjectionColumns; ++i)
    {
        int inputIdx = projectionIndices[i];
        const GPUDBMS::ColumnInfoGPU &inCol = inputColumns[inputIdx];
        GPUDBMS::ColumnInfoGPU &outCol = outputColumns[i];

        switch (inCol.type)
        {
        case GPUDBMS::DataType::INT:
        {
            const int *src = static_cast<const int *>(inCol.data);
            int *dst = static_cast<int *>(outCol.data);
            dst[row] = src[row];
            break;
        }
        case GPUDBMS::DataType::FLOAT:
        {
            const float *src = static_cast<const float *>(inCol.data);
            float *dst = static_cast<float *>(outCol.data);
            dst[row] = src[row];
            break;
        }
        case GPUDBMS::DataType::DOUBLE:
        {
            const double *src = static_cast<const double *>(inCol.data);
            double *dst = static_cast<double *>(outCol.data);
            dst[row] = src[row];
            break;
        }
        case GPUDBMS::DataType::BOOL:
        {
            const bool *src = static_cast<const bool *>(inCol.data);
            bool *dst = static_cast<bool *>(outCol.data);
            dst[row] = src[row];
            break;
        }
            // Add more cases as needed
        }
    }
}

extern "C" GPUDBMS::Table launchProjectKernel(
    const GPUDBMS::Table &inputTable,
    const std::vector<std::string> &projectColumns)
{
    const size_t rowCount = inputTable.getRowCount();
    const size_t numProjectColumns = projectColumns.size();

    if (rowCount == 0 || numProjectColumns == 0)
    {
        return inputTable; // Return original or empty table
    }

    const auto &inputCols = inputTable.getColumns();

    // Map column names to indices
    std::unordered_map<std::string, int> columnNameToIndex;
    for (size_t i = 0; i < inputCols.size(); ++i)
    {
        columnNameToIndex[inputCols[i].getName()] = static_cast<int>(i);
    }

    // Collect projection indices and metadata
    std::vector<int> h_projIndices;
    std::vector<GPUDBMS::ColumnInfoGPU> inputColInfos;
    std::vector<GPUDBMS::ColumnInfoGPU> outputColInfos;
    std::vector<void *> d_outputDataPointers;

    for (const auto &colName : projectColumns)
    {
        auto it = columnNameToIndex.find(colName);
        if (it == columnNameToIndex.end())
        {
            throw std::runtime_error("Column not found: " + colName);
        }

        int colIdx = it->second;
        h_projIndices.push_back(colIdx);

        auto inCol = inputTable.getColumnInfoGPU(colName);
        inputColInfos.push_back(inCol);

        size_t dataSize = getTypeSize(inCol.type) * rowCount;

        void *d_outputData = nullptr;
        cudaMalloc(&d_outputData, dataSize);
        d_outputDataPointers.push_back(d_outputData);

        GPUDBMS::ColumnInfoGPU outCol;
        outCol.type = inCol.type;
        outCol.data = d_outputData;
        outputColInfos.push_back(outCol);
    }

    // Allocate and copy to device
    GPUDBMS::ColumnInfoGPU *d_inputCols = nullptr;
    GPUDBMS::ColumnInfoGPU *d_outputCols = nullptr;
    int *d_projIndices = nullptr;

    cudaMalloc(&d_inputCols, inputColInfos.size() * sizeof(GPUDBMS::ColumnInfoGPU));
    cudaMemcpy(d_inputCols, inputColInfos.data(),
               inputColInfos.size() * sizeof(GPUDBMS::ColumnInfoGPU), cudaMemcpyHostToDevice);

    cudaMalloc(&d_outputCols, outputColInfos.size() * sizeof(GPUDBMS::ColumnInfoGPU));
    cudaMemcpy(d_outputCols, outputColInfos.data(),
               outputColInfos.size() * sizeof(GPUDBMS::ColumnInfoGPU), cudaMemcpyHostToDevice);

    cudaMalloc(&d_projIndices, h_projIndices.size() * sizeof(int));
    cudaMemcpy(d_projIndices, h_projIndices.data(),
               h_projIndices.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (rowCount + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch the project kernel
    projectKernel<<<numBlocks, blockSize>>>(
        static_cast<int>(rowCount),
        d_inputCols,
        static_cast<int>(inputCols.size()),
        d_projIndices,
        static_cast<int>(numProjectColumns),
        d_outputCols);

    // Record the stop event
    cudaEventRecord(stop);

    // Wait for the events to complete
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the kernel execution time
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();

    // Copy output data back to host and construct result table
    GPUDBMS::Table resultTable = inputTable.createSlicedEmptyWithSameSchema(projectColumns);

    for (size_t i = 0; i < projectColumns.size(); ++i)
    {
        size_t dataSize = getTypeSize(outputColInfos[i].type) * rowCount;

        // Create the appropriate type of vector for the data
        if (outputColInfos[i].type == GPUDBMS::DataType::INT)
        {
            std::vector<int> h_outputData(rowCount);
            cudaMemcpy(h_outputData.data(), d_outputDataPointers[i], dataSize, cudaMemcpyDeviceToHost);
            resultTable.setColumnData(i, h_outputData);
        }
        else if (outputColInfos[i].type == GPUDBMS::DataType::FLOAT)
        {
            std::vector<float> h_outputData(rowCount);
            cudaMemcpy(h_outputData.data(), d_outputDataPointers[i], dataSize, cudaMemcpyDeviceToHost);
            resultTable.setColumnData(i, h_outputData);
        }
        // Add more cases as necessary for other data types
    }

    // Cleanup
    for (void *ptr : d_outputDataPointers)
    {
        cudaFree(ptr);
    }
    cudaFree(d_inputCols);
    cudaFree(d_outputCols);
    cudaFree(d_projIndices);

    return resultTable;
}
