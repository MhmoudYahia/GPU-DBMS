#pragma once

#include "../DataHandling/Table.hpp"
#include "../DataHandling/Condition.hpp"
#include <string>

namespace GPUDBMS
{
    // Enum for Join types
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
        Join(const Table &leftTable, const Table &rightTable, const Condition &condition, JoinType joinType = JoinType::INNER);

        // Execute the join operation
        Table execute(bool useGPU = false);

    private:
        Table executeCPU();
        Table executeGPU();
        std::string getJoinTypeName() const;

        const Table &m_leftTable;
        const Table &m_rightTable;
        const Condition &m_condition;
        JoinType m_joinType;
    };
} // namespace GPUDBMS