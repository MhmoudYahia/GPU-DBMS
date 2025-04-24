#pragma once
#include "DataHandling/Table.hpp"
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <cuda_runtime.h>

namespace SQLQueryProcessor {

enum class ComparisonOperator {
    EQUAL,
    NOT_EQUAL,
    GREATER,
    GREATER_EQUAL,
    LESS,
    LESS_EQUAL
};

enum class LogicalOperator {
    AND,
    OR
};

class FilterCondition {
public:
    std::string columnName;
    ComparisonOperator compOp;
    std::string value;
    
    FilterCondition(const std::string& col, ComparisonOperator op, const std::string& val)
        : columnName(col), compOp(op), value(val) {}
};

class Filter {
public:
    Filter() = default;
    ~Filter() = default;
    
    // Set up filter conditions
    void addCondition(const FilterCondition& condition, LogicalOperator logicOp = LogicalOperator::AND);
    void clearConditions();
    
    // CPU implementation of filter operation
    std::shared_ptr<Table> executeCPU(const std::shared_ptr<Table>& inputTable);
    
    // GPU implementation of filter operation
    std::shared_ptr<Table> executeGPU(const std::shared_ptr<Table>& inputTable);
    
private:
    std::vector<FilterCondition> conditions;
    std::vector<LogicalOperator> logicalOperators;
    
    // Helper methods for evaluating conditions
    bool evaluateConditionCPU(const std::string& value, const FilterCondition& condition);
    bool evaluateRowCPU(const std::vector<std::string>& row, const std::shared_ptr<Table>& table);
};

// CUDA kernels for filter operation
__device__ bool evaluateConditionGPU(const char* value, const char* conditionValue, int compOp);
__global__ void filterKernel(const char** inputData, char** outputData, bool* resultFlags,
                           int* columnIndices, int* compOps, const char** compareValues,
                           int* logicOps, int numConditions, int numRows, int numColumns);

} // namespace SQLQueryProcessor