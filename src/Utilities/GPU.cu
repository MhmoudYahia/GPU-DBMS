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

__device__ int device_strcmp(const char *a, const char *b)
{
    while (*a && *a == *b)
    {
        a++;
        b++;
    }
    return *a - *b;
}

__device__ int device_datetime_cmp(const char *dt1, const char *dt2)
{
    // Compare year (first 4 digits)
    for (int i = 0; i < 4; i++)
    {
        if (dt1[i] != dt2[i])
            return dt1[i] - dt2[i];
    }

    // Compare month (digits 5-6)
    for (int i = 5; i < 7; i++)
    {
        if (dt1[i] != dt2[i])
            return dt1[i] - dt2[i];
    }

    // Compare day (digits 8-9)
    for (int i = 8; i < 10; i++)
    {
        if (dt1[i] != dt2[i])
            return dt1[i] - dt2[i];
    }

    // Compare hour (digits 11-12)
    for (int i = 11; i < 13; i++)
    {
        if (dt1[i] != dt2[i])
            return dt1[i] - dt2[i];
    }

    // Compare minute (digits 14-15)
    for (int i = 14; i < 16; i++)
    {
        if (dt1[i] != dt2[i])
            return dt1[i] - dt2[i];
    }

    // Compare second (digits 17-18)
    for (int i = 17; i < 19; i++)
    {
        if (dt1[i] != dt2[i])
            return dt1[i] - dt2[i];
    }

    return 0;
}

__device__ void device_strcpy(char *dest, const char *src)
{
    while ((*dest++ = *src++))
    {
    }
}