#include "../../include/DataHandling/Table.hpp"

size_t getTypeSize(GPUDBMS::DataType type);

__device__ int device_strcmp(const char *a, const char *b);

__device__ int device_datetime_cmp(const char *dt1, const char *dt2);

__device__ void device_strcpy(char *dest, const char *src);