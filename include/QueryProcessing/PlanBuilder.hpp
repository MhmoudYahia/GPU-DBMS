#pragma once
#include "DataHandling/Table.hpp"
#include "DataHandling/StorageManager.hpp"
#include "ASTProcessor.hpp"
#include "Operations/Select.hpp"
#include "Operations/Project.hpp"
#include "Operations/Join.hpp"
#include "Operations/Filter.hpp"
#include "Operations/OrderBy.hpp"
#include "Operations/Aggregator.hpp"
#include <memory>
#include <vector>
#include <string>

namespace SQLQueryProcessor
{

    class PlanBuilder
    {
    public:
        enum class ExecutionStrategy
        {
            SEQUENTIAL, // Execute operations one after another
            PIPELINED,  // Pipeline operations where possible
            PARALLEL    // Execute independent operations in parallel
        };

        PlanBuilder(StorageManager &storageManager, ExecutionStrategy strategy = ExecutionStrategy::PIPELINED);
        ~PlanBuilder() = default;

        // Build an execution plan from query information
        std::shared_ptr<Table> buildAndExecutePlan(const ASTProcessor::QueryInfo &queryInfo,
                                                   bool useGPU = true,
                                                   bool useStreaming = false);

        // Set execution strategy
        void setExecutionStrategy(ExecutionStrategy strategy);
        ExecutionStrategy getExecutionStrategy() const;

    private:
        StorageManager &storageManager;
        ExecutionStrategy executionStrategy;

        // Operation objects
        Select selectOp;
        Project projectOp;
        Join joinOp;
        Filter filterOp;
        OrderBy orderByOp;
        Aggregator aggregatorOp;

        // Plan building helper methods
        std::shared_ptr<Table> executeTableScan(const std::string &tableName);
        std::shared_ptr<Table> executeJoins(const std::vector<std::shared_ptr<Table>> &tables,
                                            const std::vector<ASTProcessor::JoinInfo> &joinInfos,
                                            bool useGPU);
        std::shared_ptr<Table> executeFilter(const std::shared_ptr<Table> &table,
                                             const std::vector<FilterCondition> &conditions,
                                             const std::vector<LogicalOperator> &logicalOps,
                                             bool useGPU);
        std::shared_ptr<Table> executeGroupByAndAggregation(const std::shared_ptr<Table> &table,
                                                            const std::vector<std::string> &groupByColumns,
                                                            const std::vector<ASTProcessor::ColumnInfo> &selectColumns,
                                                            bool useGPU);
        std::shared_ptr<Table> executeOrderBy(const std::shared_ptr<Table> &table,
                                              const std::vector<std::pair<std::string, SortOrder>> &orderByColumns,
                                              bool useGPU);
        std::shared_ptr<Table> executeProjection(const std::shared_ptr<Table> &table,
                                                 const std::vector<ASTProcessor::ColumnInfo> &selectColumns,
                                                 bool useGPU);
    };

} // namespace SQLQueryProcessor