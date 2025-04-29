<<<<<<< Updated upstream
// filepath: /media/mohamed/0B370EA20B370EA2/CMP1Materials/Forth/Second/PC/Project/GPU-DBMS/include/Operations/Filter.hpp
=======
>>>>>>> Stashed changes
#ifndef FILTER_HPP
#define FILTER_HPP

#include <vector>
#include <string>
<<<<<<< Updated upstream
#include <functional>
=======
>>>>>>> Stashed changes
#include "../DataHandling/Table.hpp"
#include "../DataHandling/Condition.hpp"

namespace GPUDBMS
{
<<<<<<< Updated upstream

    /**
     * @class Filter
     * @brief The Filter operation filters rows from a table that satisfy complex conditions.
=======
    /**
     * @class Filter
     * @brief Filters rows from a table based on a condition
>>>>>>> Stashed changes
     */
    class Filter
    {
    public:
        /**
<<<<<<< Updated upstream
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
=======
         * @brief Construct a new Filter operation
         * 
         * @param inputTable The input table to filter
         * @param condition The condition to apply for filtering
         */
        Filter(const Table &inputTable, const Condition &condition);

        /**
         * @brief Execute the filter operation using GPU acceleration
         * 
         * @return Table The filtered result table
>>>>>>> Stashed changes
         */
        Table execute();

        /**
<<<<<<< Updated upstream
         * @brief Execute the filter operation on CPU (fallback).
         *
         * @return Table The resulting table containing only rows that satisfy the complex condition.
=======
         * @brief Execute the filter operation on CPU (fallback)
         * 
         * @return Table The filtered result table
>>>>>>> Stashed changes
         */
        Table executeCPU();

    private:
        const Table &m_inputTable;
<<<<<<< Updated upstream
        std::vector<std::unique_ptr<Condition>> m_conditions;

        bool m_isAnd; // true for AND logic, false for OR logic
=======
        const Condition &m_condition;
>>>>>>> Stashed changes
    };

} // namespace GPUDBMS

#endif // FILTER_HPP