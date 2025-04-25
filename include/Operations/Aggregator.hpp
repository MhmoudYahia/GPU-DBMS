#pragma once
#include "DataHandling/Table.hpp"
#include <memory>
#include <vector>
#include <string>
#include <cuda_runtime.h>

namespace SQLQueryProcessor
{

    enum class AggregateFunction
    {
        COUNT,
        SUM,
        AVG,
        MIN,
        MAX
    };

    class Aggregator
    {
    public:
        Aggregator() = default;
        ~Aggregator() = default;

        // Set up aggregation parameters
        void addAggregation(AggregateFunction func, const std::string &columnName, const std::string &alias = "");
        void addGroupByColumn(const std::string &columnName);
        void clearAggregations();
        void clearGroupByColumns();

        // CPU implementation of aggregation operation
        std::shared_ptr<Table> executeCPU(const std::shared_ptr<Table> &inputTable);

        // GPU implementation of aggregation operation
        std::shared_ptr<Table> executeGPU(const std::shared_ptr<Table> &inputTable);

    private:
        struct AggregateInfo
        {
            AggregateFunction function;
            std::string columnName;
            std::string alias;

            AggregateInfo(AggregateFunction func, const std::string &col, const std::string &als = "")
                : function(func), columnName(col), alias(als.empty() ? generateDefaultAlias(func, col) : als) {}

            static std::string generateDefaultAlias(AggregateFunction func, const std::string &col);
        };

        std::vector<AggregateInfo> aggregations;
        std::vector<std::string> groupByColumns;

        // Helper methods for CPU implementation
        std::string applyAggregateFunction(AggregateFunction func, const std::vector<std::string> &values);
        double parseNumeric(const std::string &value);
    };

    // CUDA kernels for aggregation operations
    __global__ void countKernel(const char **inputData, int *results, int *groupIndices,
                                int numRows, int numGroups);

    __global__ void sumKernel(const char **inputData, double *results, int *groupIndices,
                              int columnIndex, int numRows, int numGroups);

    __global__ void minMaxKernel(const char **inputData, double *results, int *groupIndices,
                                 int columnIndex, int numRows, int numGroups, bool isMin);

} // namespace SQLQueryProcessor