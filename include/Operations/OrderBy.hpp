#pragma once
#include "DataHandling/Table.hpp"
#include <memory>
#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace SQLQueryProcessor
{

    enum class SortOrder
    {
        ASCENDING,
        DESCENDING
    };

    enum class SortType
    {
        NUMERIC,
        TEXT,
        DATETIME
    };

    class OrderBy
    {
    public:
        OrderBy() = default;
        ~OrderBy() = default;

        // Set sort parameters
        void addSortKey(const std::string &columnName, SortOrder order = SortOrder::ASCENDING);
        void clearSortKeys();

        // CPU implementation of order by operation
        std::shared_ptr<Table> executeCPU(const std::shared_ptr<Table> &inputTable);

        // GPU implementation of order by operation
        std::shared_ptr<Table> executeGPU(const std::shared_ptr<Table> &inputTable);

    private:
        struct SortKey
        {
            std::string columnName;
            SortOrder order;
            SortType type;

            SortKey(const std::string &name, SortOrder ord, SortType typ = SortType::TEXT)
                : columnName(name), order(ord), type(typ) {}
        };

        std::vector<SortKey> sortKeys;

        // Helper methods
        SortType determineSortType(const std::shared_ptr<Table> &table, const std::string &columnName);

        // CPU sorting implementation
        void quickSortCPU(std::vector<std::vector<std::string>> &data, int low, int high,
                          const std::vector<int> &keyIndices, const std::vector<SortOrder> &orders,
                          const std::vector<SortType> &types);

        int partitionCPU(std::vector<std::vector<std::string>> &data, int low, int high,
                         const std::vector<int> &keyIndices, const std::vector<SortOrder> &orders,
                         const std::vector<SortType> &types);

        int compareValues(const std::string &val1, const std::string &val2,
                          SortType type, SortOrder order);
    };

    // CUDA kernel for bitonic sort
    __global__ void bitonicSortKernel(char **data, int *keyIndices, int *sortOrders,
                                      int *sortTypes, int numRows, int numColumns,
                                      int numKeys, int step, int stage);

} // namespace SQLQueryProcessor