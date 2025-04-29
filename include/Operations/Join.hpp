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
        INNER,
        LEFT,
        RIGHT,
        FULL
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
         * @param type The type of join (defaults to INNER)
         */
        Join(const Table &leftTable, const Table &rightTable, const Condition &condition, JoinType type = JoinType::INNER);

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