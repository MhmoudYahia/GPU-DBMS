#ifndef SELECT_HPP
#define SELECT_HPP

#include <vector>
#include <string>
#include <functional>
#include "../DataHandling/Table.hpp"
#include "../DataHandling/Condition.hpp"
#include "../Operations/SelectGPU.cuh"

namespace GPUDBMS
{

    /**
     * @class Select
     * @brief The Select operation filters rows from a table that satisfy a given condition.
     */
    class Select
    {
    public:
        /**
         * @brief Construct a new Select operation.
         *
         * @param inputTable The input table to select rows from.
         * @param condition The condition that rows must satisfy to be selected.
         */
        Select(const Table &inputTable, const Condition &condition);


        /**
         * @brief Execute the selection operation on GPU.
         *
         * @return Table The resulting table containing only rows that satisfy the condition.
         */
        Table execute(bool useGPU = false);

        /**
         * @brief Execute the selection operation on CPU (fallback).
         *
         * @return Table The resulting table containing only rows that satisfy the condition.
         */
        Table executeCPU();

        Table executeGPU();

    private:
        const Table &m_inputTable;
        const Condition &m_condition;
    };

    // /**
    //  * @class Project
    //  * @brief The Project operation selects certain columns from a table.
    //  */
    // class Project
    // {
    // public:
    //     /**
    //      * @brief Construct a new Project operation.
    //      *
    //      * @param inputTable The input table to project columns from.
    //      * @param columnNames Names of columns to keep in the result.
    //      */
    //     Project(const Table &inputTable, const std::vector<std::string> &columnNames);

    //     /**
    //      * @brief Execute the projection operation on GPU.
    //      *
    //      * @return Table The resulting table containing only the specified columns.
    //      */
    //     Table execute(bool useGPU = false);

    //     /**
    //      * @brief Execute the projection operation on CPU (fallback).
    //      *
    //      * @return Table The resulting table containing only the specified columns.
    //      */
    //     Table executeCPU();

    // private:
    //     const Table &m_inputTable;
    //     std::vector<std::string> m_columnNames;
    //     std::vector<int> m_columnIndices; // Indices of columns to project

    //     /**
    //      * @brief Convert column names to their corresponding indices in the input table.
    //      */
    //     void resolveColumnIndices();
    // };

} // namespace GPUDBMS

#endif // SELECT_HPP