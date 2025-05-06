#ifndef ORDERBY_GPU_CUH
#define ORDERBY_GPU_CUH
#include <cuda_runtime.h>
#include "../../include/DataHandling/Condition.hpp"
#include "../../include/DataHandling/Table.hpp"
#include "../../include/Operations/SelectGPU.cuh"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>



enum class SortOrder
{
    ASC,
    DESC
};

#endif