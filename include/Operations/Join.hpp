#ifndef JOIN_HPP
#define JOIN_HPP

#include <vector>
#include <string>
#include "../DataHandling/Table.hpp"
#include "../DataHandling/Condition.hpp"

namespace GPUDBMS
{
    /**
     * @enum JoinType
     * @brief Types of join operations
     */
    enum class JoinType
    {
        INNER,   // Only matching rows
        LEFT,    // All rows from left table, matching rows from right
        RIGHT,   // All rows from right table, matching rows from left
        FULL     // All rows from both tables
    };

    /**
     * @class Join
     * @brief Joins two tables based on a join condition
     */
    class Join
    {
    public:
        /**
         * @brief Construct a Join operation
         * 
         * @param leftTable The left table in the join
         * @param rightTable The right table in the join
         * @param condition The join condition
         * @param joinType The type of join (defaults to INNER)
         */
        Join(const Table &leftTable, const Table &rightTable, 
             const Condition &condition, JoinType joinType = JoinType::INNER);

        /**
         * @brief Execute the join operation using GPU acceleration
         * 
         * @return Table The joined result table
         */
        Table execute();

        /**
         * @brief Execute the join operation on CPU (fallback)
         * 
         * @return Table The joined result table
         */
        Table executeCPU();

    private:
        const Table &m_leftTable;
        const Table &m_rightTable;
        const Condition &m_condition;
        JoinType m_joinType;

        /**
         * @brief Create the schema for the result table
         * 
         * @return std::vector<Column> Schema for the result table
         */
        std::vector<Column> createResultSchema() const;
    };

} // namespace GPUDBMS

#endif // JOIN_HPP