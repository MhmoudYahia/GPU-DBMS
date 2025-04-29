#ifndef AGGREGATOR_HPP
#define AGGREGATOR_HPP

#include <vector>
#include <string>
#include <optional>
#include "../DataHandling/Table.hpp"

namespace GPUDBMS
{
    /**
     * @enum AggregateFunction
     * @brief Supported aggregate functions
     */
    enum class AggregateFunction
    {
        COUNT,
        SUM,
        AVG,
        MIN,
        MAX
    };

    /**
     * @class Aggregation
     * @brief Represents a single aggregation operation
     */
    struct Aggregation
    {
        AggregateFunction function;
        std::string columnName;
        std::string resultName;  // Alias for the result column

        Aggregation(AggregateFunction func, const std::string& col, const std::string& alias = "")
            : function(func), columnName(col), 
              resultName(alias.empty() ? getDefaultAlias(func, col) : alias) {}

    private:
        static std::string getDefaultAlias(AggregateFunction func, const std::string& col) {
            std::string prefix;
            switch (func) {
                case AggregateFunction::COUNT: prefix = "count_"; break;
                case AggregateFunction::SUM: prefix = "sum_"; break;
                case AggregateFunction::AVG: prefix = "avg_"; break;
                case AggregateFunction::MIN: prefix = "min_"; break;
                case AggregateFunction::MAX: prefix = "max_"; break;
            }
            return prefix + col;
        }
    };

    /**
     * @class Aggregator
     * @brief Performs aggregation operations on a table
     */
    class Aggregator
    {
    public:
        /**
         * @brief Construct an Aggregator with a single aggregation function
         * 
         * @param inputTable The input table
         * @param function The aggregate function to apply
         * @param columnName The column to aggregate
         * @param groupByColumn Optional column to group by
         * @param alias Optional alias for the result column
         */
        Aggregator(const Table& inputTable, 
                  AggregateFunction function, 
                  const std::string& columnName,
                  const std::optional<std::string>& groupByColumn = std::nullopt,
                  const std::string& alias = "");
        
        /**
         * @brief Construct an Aggregator with multiple aggregation functions
         * 
         * @param inputTable The input table
         * @param aggregations Vector of aggregation operations
         * @param groupByColumn Optional column to group by
         */
        Aggregator(const Table& inputTable, 
                  const std::vector<Aggregation>& aggregations,
                  const std::optional<std::string>& groupByColumn = std::nullopt);
        
        /**
         * @brief Execute the aggregation operation
         * 
         * @return Table The result table with aggregated values
         */
        Table execute();
        
        /**
         * @brief Execute the aggregation on CPU
         * 
         * @return Table The result table with aggregated values
         */
        Table executeCPU();
        
    private:
        const Table& m_inputTable;
        std::vector<Aggregation> m_aggregations;
        std::optional<std::string> m_groupByColumn;
        std::optional<int> m_groupByColumnIndex;
        
        /**
         * @brief Resolve the group by column index
         */
        void resolveGroupByColumnIndex();
        
        /**
         * @brief Perform a specific aggregation function on a column
         */
        template <typename T>
        T aggregateValues(const std::vector<T>& values, AggregateFunction function) const;
    };

} // namespace GPUDBMS

#endif // AGGREGATOR_HPP