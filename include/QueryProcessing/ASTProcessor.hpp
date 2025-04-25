#pragma once
#include "DataHandling/Table.hpp"
#include "DataHandling/StorageManager.hpp"
#include "Operations/Filter.hpp"
#include "Operations/OrderBy.hpp"
#include "Operations/Aggregator.hpp"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace SQLQueryProcessor
{

    class ASTProcessor
    {
    public:
        struct JoinInfo
        {
            std::string leftTable;
            std::string rightTable;
            std::string leftColumn;
            std::string rightColumn;
        };

        struct ColumnInfo
        {
            std::string column;
            std::string table;
            std::string alias;
            bool isAggregation;
            AggregateFunction aggrFunc;

            ColumnInfo(const std::string &col = "", const std::string &tab = "",
                       const std::string &als = "", bool isAggr = false,
                       AggregateFunction func = AggregateFunction::COUNT)
                : column(col), table(tab), alias(als), isAggregation(isAggr), aggrFunc(func) {}
        };

        struct QueryInfo
        {
            std::vector<ColumnInfo> selectColumns;
            std::vector<std::string> fromTables;
            std::vector<JoinInfo> joins;
            std::vector<FilterCondition> whereConditions;
            std::vector<LogicalOperator> whereLogicalOps;
            std::vector<std::string> groupByColumns;
            std::vector<std::pair<std::string, SortOrder>> orderByColumns;
            std::string subQuery;
            bool hasSubquery;

            QueryInfo() : hasSubquery(false) {}
        };

        ASTProcessor(StorageManager &storageManager);
        ~ASTProcessor() = default;

        // Process a SQL query and extract structured information
        QueryInfo processQuery(const std::string &query);

    private:
        StorageManager &storageManager;

        // Helper methods for SQL query parsing
        std::string extractSubquery(const std::string &query);
        std::vector<ColumnInfo> parseSelectClause(const std::string &selectClause);
        std::vector<std::string> parseFromClause(const std::string &fromClause);
        std::vector<JoinInfo> parseJoinConditions(const std::string &whereClause, const std::vector<std::string> &tables);
        void parseWhereConditions(const std::string &whereClause, QueryInfo &queryInfo);
        std::vector<std::string> parseGroupByClause(const std::string &groupByClause);
        std::vector<std::pair<std::string, SortOrder>> parseOrderByClause(const std::string &orderByClause);

        // Splitting and tokenizing helpers
        std::vector<std::string> tokenizeClause(const std::string &clause, const std::string &delimiter);
        std::string extractClause(const std::string &query, const std::string &clauseStart, const std::string &clauseEnd = "");
    };

} // namespace SQLQueryProcessor