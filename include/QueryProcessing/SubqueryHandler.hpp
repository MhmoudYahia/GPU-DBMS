#pragma once
#include "DataHandling/Table.hpp"
#include "QueryExecutor.hpp"
#include <memory>
#include <string>

namespace SQLQueryProcessor
{

    class SubqueryHandler
    {
    public:
        SubqueryHandler(QueryExecutor &queryExecutor);
        ~SubqueryHandler() = default;

        // Extract and process a subquery
        std::shared_ptr<Table> processSubquery(const std::string &fullQuery, size_t &subqueryStart, size_t &subqueryEnd);

        // Check if a query contains a subquery
        bool hasSubquery(const std::string &query, size_t &start, size_t &end);

        // Replace a subquery with a temporary table reference
        std::string replaceSubqueryWithTemp(const std::string &query, const std::string &tempTableName,
                                            size_t subqueryStart, size_t subqueryEnd);

    private:
        QueryExecutor &queryExecutor;

        // Helper methods
        std::string extractSubquery(const std::string &query, size_t start, size_t end);
        int findMatchingParenthesis(const std::string &query, size_t openPos);
    };

} // namespace SQLQueryProcessor