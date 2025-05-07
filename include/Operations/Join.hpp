#pragma once

#include "../DataHandling/Table.hpp"
#include "../DataHandling/Condition.hpp"

namespace GPUDBMS
{

    enum class JoinType
    {
        INNER,
        LEFT,
        RIGHT,
        FULL
    };

    class Join
    {
    public:
        Join(const Table &leftTable, const Table &rightTable,
             const Condition &condition, JoinType joinType = JoinType::INNER);

        Table execute(bool useGPU = false);
        Table executeCPU();
        Table executeGPU();

    private:
        const Table &m_leftTable;
        const Table &m_rightTable;
        const Condition &m_condition;
        JoinType m_joinType;

        std::vector<Column> createResultSchema() const;
    };

} // namespace GPUDBMS