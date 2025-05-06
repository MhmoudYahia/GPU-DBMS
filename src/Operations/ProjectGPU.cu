#include "../../include/Operations/ProjectGPU.cuh"

template <typename T>
__global__ void projectKernel(const T *__restrict__ inputCols,
                              T *__restrict__ outputCols,
                              const int *selectedColIndices,
                              int numSelectedCols,
                              int numRows,
                              int numCols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y; // one thread block per selected column

    if (row >= numRows || col >= numSelectedCols)
        return;

    int colIdx = selectedColIndices[col]; // which column in input to project
    outputCols[col * numRows + row] = inputCols[colIdx * numRows + row];
}

template <typename T>
void launchProjectKernelCUDA(const T *h_inputCols,
                             T *h_outputCols,
                             const std::vector<int> &selectedColIndices,
                             int numRows,
                             int numCols)
{
    int numSelectedCols = selectedColIndices.size();
    size_t inputSize = numRows * numCols * sizeof(T);
    size_t outputSize = numRows * numSelectedCols * sizeof(T);
    size_t indicesSize = numSelectedCols * sizeof(int);

    // Allocate device memory
    T *d_inputCols;
    T *d_outputCols;
    int *d_selectedColIndices;

    cudaMalloc(&d_inputCols, inputSize);
    cudaMalloc(&d_outputCols, outputSize);
    cudaMalloc(&d_selectedColIndices, indicesSize);

    cudaMemcpy(d_inputCols, h_inputCols, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_selectedColIndices, selectedColIndices.data(), indicesSize, cudaMemcpyHostToDevice);

    dim3 blockDim(256);
    dim3 gridDim((numRows + blockDim.x - 1) / blockDim.x, numSelectedCols); // 1 block per col

    projectKernel<T><<<gridDim, blockDim>>>(
        d_inputCols, d_outputCols, d_selectedColIndices,
        numSelectedCols, numRows, numCols);

    cudaMemcpy(h_outputCols, d_outputCols, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_inputCols);
    cudaFree(d_outputCols);
    cudaFree(d_selectedColIndices);
}

void projectTableCUDA(const void *inputCols,
                      void *outputCols,
                      const std::vector<int> &selectedColIndices,
                      int numRows,
                      int numCols,
                      GPUDBMS::DataType type)
{
    switch (type)
    {
    case GPUDBMS::DataType::INT:
        launchProjectKernelCUDA<int>(
            static_cast<const int *>(inputCols),
            static_cast<int *>(outputCols),
            selectedColIndices,
            numRows, numCols);
        break;

    case GPUDBMS::DataType::FLOAT:
        launchProjectKernelCUDA<float>(
            static_cast<const float *>(inputCols),
            static_cast<float *>(outputCols),
            selectedColIndices,
            numRows, numCols);
        break;

    case GPUDBMS::DataType::DOUBLE:
        launchProjectKernelCUDA<double>(
            static_cast<const double *>(inputCols),
            static_cast<double *>(outputCols),
            selectedColIndices,
            numRows, numCols);
        break;

    default:
        std::cerr << "Unsupported data type!" << std::endl;
        break;
    }
}