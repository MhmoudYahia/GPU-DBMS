// filepath: /media/mohamed/0B370EA20B370EA2/CMP1Materials/Forth/Second/PC/Project/GPU-DBMS/include/Operations/Filter.hpp
#ifndef FILTER_HPP
#define FILTER_HPP

#include <vector>
#include <string>
#include <functional>
#include "../DataHandling/Table.hpp"
#include "../DataHandling/Condition.hpp"

namespace GPUDBMS
{

    /**
     * @class Filter
     * @brief The Filter operation filters rows from a table that satisfy complex conditions.
     */
    class Filter
    {
    public:
        /**
         * @brief Construct a new Filter operation.
         *
         * @param inputTable The input table to filter rows from.
         * @param conditions Vector of conditions that rows must satisfy (combined with AND/OR logic).
         * @param isAnd True if conditions are combined with AND, false for OR logic.
         */
        Filter(const Table &inputTable, const std::vector<std::unique_ptr<Condition>> &conditions, bool isAnd);

        /**
         * @brief Execute the filter operation on GPU.
         *
         * @return Table The resulting table containing only rows that satisfy the complex condition.
         */
        Table execute();

        /**
         * @brief Execute the filter operation on CPU (fallback).
         *
         * @return Table The resulting table containing only rows that satisfy the complex condition.
         */
        Table executeCPU();

    private:
        const Table &m_inputTable;
        std::vector<std::unique_ptr<Condition>> m_conditions;

        bool m_isAnd; // true for AND logic, false for OR logic
    };

} // namespace GPUDBMS

#endif // FILTER_HPP