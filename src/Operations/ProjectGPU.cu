#include "../../include/Operations/ProjectGPU.cuh"

__global__ void projectKernel(
    int numRows,
    const GPUDBMS::ColumnInfoGPU *inputColumns,
    int numInputColumns,
    const int *projectionIndices,
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

        if (inCol.data == nullptr || outCol.data == nullptr)
            continue;

        // Ensure output column has proper count
        outCol.count = numRows;

        switch (inCol.type)
        {
        case GPUDBMS::DataType::INT:
        {
            const int *src = static_cast<const int *>(inCol.data);
            int *dst = static_cast<int *>(outCol.data);
            printf("Copying int data from input column %d to output column %d\n", inputIdx, i);
            printf("Input column count: %d, Output column count: %d\n", inCol.count, outCol.count);
            printf("Row: %d, Input value: %d\n", row, (row < inCol.count) ? src[row] : 0);
            dst[row] = (row < inCol.count) ? src[row] : 0;
            break;
        }
        case GPUDBMS::DataType::FLOAT:
        {
            const float *src = static_cast<const float *>(inCol.data);
            float *dst = static_cast<float *>(outCol.data);
            dst[row] = (row < inCol.count) ? src[row] : 0.0f;
            break;
        }
        case GPUDBMS::DataType::DOUBLE:
        {
            const double *src = static_cast<const double *>(inCol.data);
            double *dst = static_cast<double *>(outCol.data);
            dst[row] = (row < inCol.count) ? src[row] : 0.0;
            break;
        }
        case GPUDBMS::DataType::BOOL:
        {
            const bool *src = static_cast<const bool *>(inCol.data);
            bool *dst = static_cast<bool *>(outCol.data);
            dst[row] = (row < inCol.count) ? src[row] : false;
            break;
        }
        case GPUDBMS::DataType::STRING:
        case GPUDBMS::DataType::VARCHAR:
        case GPUDBMS::DataType::DATE:
        case GPUDBMS::DataType::DATETIME:
        {
            const char *src = static_cast<const char *>(inCol.data);
            char *dst = static_cast<char *>(outCol.data);

            // Ensure strides are properly set
            size_t inStride = inCol.stride > 0 ? inCol.stride : 256; // Default stride if not set
            size_t outStride = outCol.stride > 0 ? outCol.stride : inStride;

            if (row < inCol.count)
            {
                const char *srcStr = src + row * inStride;
                char *dstStr = dst + row * outStride;

                // Copy string (assuming null-terminated)
                int pos = 0;
                while (pos < outStride && srcStr[pos] != '\0')
                {
                    dstStr[pos] = srcStr[pos];
                    pos++;
                }
                if (pos < outStride)
                {
                    dstStr[pos] = '\0';
                }
            }
            else
            {
                // Set to empty string for out-of-bounds rows
                char *dstStr = dst + row * outStride;
                if (outStride > 0)
                {
                    dstStr[0] = '\0';
                }
            }
            break;
        }
        default:
            break;
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

    // First pass: collect metadata and allocate output buffers
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
        outCol.count = rowCount;      // Set the correct row count
        outCol.stride = inCol.stride; // Copy the stride from input
        outputColInfos.push_back(outCol);

    }

    // Second pass: copy input data to device
    std::vector<GPUDBMS::ColumnInfoGPU> d_inputColInfos = inputColInfos;
    for (size_t i = 0; i < inputColInfos.size(); ++i)
    {
        const auto &colInfo = inputColInfos[i];
        size_t dataSize = getTypeSize(colInfo.type) * rowCount;

        // Allocate device memory for input column data
        void *d_inputData = nullptr;
        cudaMalloc(&d_inputData, dataSize);

        // Copy data from host to device
        cudaMemcpy(d_inputData, colInfo.data, dataSize, cudaMemcpyHostToDevice);

        // Update the column info with device pointer
        d_inputColInfos[i].data = d_inputData;
    }

    // Allocate and copy metadata to device
    GPUDBMS::ColumnInfoGPU *d_inputCols = nullptr;
    GPUDBMS::ColumnInfoGPU *d_outputCols = nullptr;
    int *d_projIndices = nullptr;

    cudaMalloc(&d_inputCols, d_inputColInfos.size() * sizeof(GPUDBMS::ColumnInfoGPU));
    cudaMemcpy(d_inputCols, d_inputColInfos.data(),
               d_inputColInfos.size() * sizeof(GPUDBMS::ColumnInfoGPU), cudaMemcpyHostToDevice);

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

    cudaEventRecord(start);
    projectKernel<<<numBlocks, blockSize>>>(
        rowCount,
        d_inputCols,
        inputCols.size(),
        d_projIndices,
        numProjectColumns,
        d_outputCols);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceSynchronize();

    // Copy output data back to host and construct result table
    GPUDBMS::Table resultTable = inputTable.createSlicedEmptyWithSameSchema(projectColumns);

    for (size_t i = 0; i < projectColumns.size(); ++i)
    {
        size_t dataSize = getTypeSize(outputColInfos[i].type) * rowCount;

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
        else if (outputColInfos[i].type == GPUDBMS::DataType::DOUBLE)
        {
            std::vector<double> h_outputData(rowCount);
            cudaMemcpy(h_outputData.data(), d_outputDataPointers[i], dataSize, cudaMemcpyDeviceToHost);
            resultTable.setColumnData(i, h_outputData);
        }
        else if (outputColInfos[i].type == GPUDBMS::DataType::BOOL)
        {
            std::vector<char> temp_buffer(rowCount);
            cudaMemcpy(temp_buffer.data(), d_outputDataPointers[i], dataSize, cudaMemcpyDeviceToHost);

            std::vector<bool> h_outputData(rowCount);
            for (size_t j = 0; j < rowCount; ++j)
            {
                h_outputData[j] = temp_buffer[j] != 0;
            }
            resultTable.setColumnData(i, h_outputData);
        }
        else if (outputColInfos[i].type == GPUDBMS::DataType::STRING ||
                 outputColInfos[i].type == GPUDBMS::DataType::VARCHAR ||
                 outputColInfos[i].type == GPUDBMS::DataType::DATE ||
                 outputColInfos[i].type == GPUDBMS::DataType::DATETIME)
        {
            std::vector<char> h_outputData(dataSize);
            cudaMemcpy(h_outputData.data(), d_outputDataPointers[i], dataSize, cudaMemcpyDeviceToHost);
            resultTable.setColumnData(i, h_outputData);
        }
    }

    // Cleanup
    for (void *ptr : d_outputDataPointers)
    {
        cudaFree(ptr);
    }

    // Free input data copies on device
    for (auto &colInfo : d_inputColInfos)
    {
        cudaFree(colInfo.data);
    }

    cudaFree(d_inputCols);
    cudaFree(d_outputCols);
    cudaFree(d_projIndices);

    cudaDeviceReset();

    return resultTable;
}
