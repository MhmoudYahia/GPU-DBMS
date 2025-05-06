#include "../../include/Utilities/GPU.cuh"

// Helper function to get size of a data type
size_t getTypeSize(GPUDBMS::DataType type)
{
    switch (type)
    {
    case GPUDBMS::DataType::INT:
        return sizeof(int);
    case GPUDBMS::DataType::FLOAT:
        return sizeof(float);
    case GPUDBMS::DataType::BOOL:
        return sizeof(bool);
    case GPUDBMS::DataType::DOUBLE:
        return sizeof(double);
    case GPUDBMS::DataType::STRING:
    case GPUDBMS::DataType::VARCHAR:
        return 256; // Adjust based on your string storage
    default:
        return 0;
    }
}