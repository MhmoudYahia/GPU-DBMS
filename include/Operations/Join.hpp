#ifndef JOIN_HPP
#define JOIN_HPP

#include <vector>
#include <string>
#include <functional>
#include <unordered_set>
#include "../DataHandling/Table.hpp"
#include "../DataHandling/Condition.hpp"
#include "../Operations/JoinGPU.cuh"

namespace GPUDBMS
{

    enum JoinType
    {
        INNER,
        LEFT,
        RIGHT,
        FULL

    };

    /**
     * @class Join
     * @brief The Join operation combines rows from two tables based on a related column.
     */
    class Join
    {
    public:
        /**
         * @brief Construct a new Join operation.
         *
         * @param leftTable The left table in the join operation.
         * @param rightTable The right table in the join operation.
         * @param condition The join condition to determine matching rows.
         */
        Join(const Table &leftTable, const Table &rightTable, const Condition &condition);

        /**
         * @brief Execute the join operation on either CPU or GPU based on parameter.
         *
         * @param useGPU Flag to determine whether to use GPU or CPU implementation.
         * @return Table The resulting joined table.
         */
        Table execute(bool useGPU = false);

        /**
         * @brief Execute the join operation on CPU (fallback).
         *
         * @return Table The resulting joined table.
         */
        Table executeCPU();

        /**
         * @brief Execute the join operation on GPU.
         *
         * @return Table The resulting joined table.
         */
        Table executeGPU();

    private:
        const Table &m_leftTable;
        const Table &m_rightTable;
        const Condition &m_condition;

        /**
         * @brief Helper function to create the schema for the resulting joined table.
         *
         * @return vector<Column> The column schemas for the joined table.
         */
        std::vector<Column> createJoinedSchema() const;
    };

} // namespace GPUDBMS

#endif // JOIN_HPP