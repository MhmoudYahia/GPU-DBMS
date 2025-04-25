#include "../../include/Operations/Select.hpp"
#include <iostream>
#include <vector>
#include <algorithm>

namespace GPUDBMS
{

    Project::Project(const Table &inputTable, const std::vector<std::string> &columnNames)
        : m_inputTable(inputTable), m_columnNames(columnNames)
    {
        resolveColumnIndices();
    }

    void Project::resolveColumnIndices()
    {
        m_columnIndices.clear();
        for (const auto &columnName : m_columnNames)
        {
            int index = m_inputTable.getColumnIndex(columnName);
            if (index < 0)
            {
                throw std::runtime_error("Column not found: " + columnName);
            }
            m_columnIndices.push_back(index);
        }
    }

    Table Project::execute()
    {
        // For now, just call CPU implementation
        // TODO: Implement GPU version
        return executeCPU();
    }

    Table Project::executeCPU()
    {
        // Create a new table with only the selected columns
        std::vector<Column> resultColumns;
        for (int index : m_columnIndices)
        {
            resultColumns.push_back(m_inputTable.getColumns()[index]);
        }

        Table resultTable(resultColumns);

        // Copy data from input table to result table
        const size_t rowCount = m_inputTable.getRowCount();

        // In a real implementation, we would copy the data column by column
        // while respecting the type of each column

        return resultTable;
    }

} // namespace GPUDBMS