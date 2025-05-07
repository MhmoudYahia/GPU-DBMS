#include "../../include/Operations/JoinGPU.cuh"
#include <iostream>
#include <vector>

template <typename T>
__global__ void joinKernel(
    const T* leftTable, int leftCols, int leftRows,
    const T* rightTable, int rightCols, int rightRows,
    T* outputTable, int* outputCount,
    int leftJoinCol, int rightJoinCol, int typeEnum
) {
    int leftRow = blockIdx.y * blockDim.y + threadIdx.y;
    int rightRow = blockIdx.x * blockDim.x + threadIdx.x;

    if (leftRow >= leftRows || rightRow >= rightRows) return;

    T leftJoinVal = leftTable[leftJoinCol * leftRows + leftRow];
    T rightJoinVal = rightTable[rightJoinCol * rightRows + rightRow];

                   
    if (m_condition.evaluate(typeEnum, leftJoinVal, rightJoinVal)) {
        int index = atomicAdd(outputCount, 1);
        // Write left row
        for (int i = 0; i < leftCols; ++i)
            outputTable[index * (leftCols + rightCols) + i] = leftTable[i * leftRows + leftRow];

        // Write right row
        for (int i = 0; i < rightCols; ++i)
            outputTable[index * (leftCols + rightCols) + leftCols + i] = rightTable[i * rightRows + rightRow];
    }
}


template <typename T>
void launchJoinKernel(
    Table* resultTable,int leftCols, int leftRows,
    int rightCols, int rightRows, int typeEnum
){
           
    std::vector<T> h_left(leftCols * leftRows);
    std::vector<T> h_right(rightCols * rightRows);

    for (int c = 0; c < leftCols; ++c)
        for (int r = 0; r < leftRows; ++r)
            h_left[c * leftRows + r] = m_leftTable.getValue<T>(c, r);

    for (int c = 0; c < rightCols; ++c)
        for (int r = 0; r < rightRows; ++r)
            h_right[c * rightRows + r] = m_rightTable.getValue<T>(c, r);

    T *d_left, *d_right, *d_output;
    int *d_count;
    int maxOutputSize = leftRows * rightRows;
    cudaMalloc(&d_left, sizeof(T) * h_left.size());
    cudaMalloc(&d_right, sizeof(T) * h_right.size());
    cudaMalloc(&d_output, sizeof(T) * maxOutputSize * (leftCols + rightCols));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_left, h_left.data(), sizeof(T) * h_left.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, h_right.data(), sizeof(T) * h_right.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(int));

    dim3 blockDim(16, 16);
    dim3 gridDim((rightRows + 15) / 16, (leftRows + 15) / 16);
    joinKernel<T><<<gridDim, blockDim>>>(
        d_left, leftCols, leftRows,
        d_right, rightCols, rightRows,
        d_output, d_count,
        leftJoinCol, rightJoinCol
    );

    std::vector<T> h_output(maxOutputSize * (leftCols + rightCols));
    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output.data(), d_output, sizeof(T) * h_count * (leftCols + rightCols), cudaMemcpyDeviceToHost);

    for (int i = 0; i < h_count; ++i) {
        for (int j = 0; j < leftCols + rightCols; ++j)
            resultTable.appendValue<T>(j, h_output[i * (leftCols + rightCols) + j]);
        resultTable.finalizeRow();
    }

    cudaFree(d_left);    
    cudaFree(d_right);
    cudaFree(d_output);
    cudaFree(d_count); 
    }


