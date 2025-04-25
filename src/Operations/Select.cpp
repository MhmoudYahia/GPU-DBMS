#include "../../include/Operations/Select.hpp"
#include <iostream>
#include <vector>
#include <algorithm>

namespace GPUDBMS
{

    // Select Implementation
    Select::Select(const Table &inputTable, const Condition &condition)
        : m_inputTable(inputTable), m_condition(condition)
    {
    }

    Table Select::execute()
    {
        // For now, just call CPU implementation
        // TODO: Implement GPU version
        return executeCPU();
    }

    Table Select::executeCPU()
    {
        // std::cout << m_inputTable.getColumnData(0).size() << std::endl;
        // std::cout << m_inputTable.getColumnData(1).size() << std::endl;
        // std::cout << m_inputTable.getColumnData(2).size() << std::endl;
        // std::cout << m_inputTable.getColumnData(3).size() << std::endl;
        // std::cout << m_inputTable.getColumnData(4).size() << std::endl;

        Table resultTable = m_inputTable.createEmptyWithSameSchema();
        const size_t rowCount = m_inputTable.getRowCount();
        const size_t colCount = m_inputTable.getColumnCount();


        std::unordered_map<std::string, int> columnNameToIndex;
        for (size_t i = 0; i < colCount; ++i)
        {
            columnNameToIndex[m_inputTable.getColumns()[i].getName()] = static_cast<int>(i);
        }

        // For each row in the input table
        for (size_t row = 0; row < rowCount; ++row)
        {
            // Extract current row data
            std::vector<std::string> rowData(colCount);
            for (size_t col = 0; col < colCount; ++col)
            {
                // Get column data based on column type
                const auto &column = m_inputTable.getColumns()[col];
                switch (column.getType())
                {
                case DataType::INT:
                    rowData[col] = std::to_string(m_inputTable.getIntValue(col, row));
                    break;
                case DataType::FLOAT:
                    rowData[col] = std::to_string(m_inputTable.getFloatValue(col, row));
                    break;
                case DataType::STRING:
                    rowData[col] = m_inputTable.getStringValue(col, row);
                    break;
                case DataType::VARCHAR:
                    rowData[col] = m_inputTable.getStringValue(col, row);
                    break;
                case DataType::DOUBLE:
                    rowData[col] = std::to_string(m_inputTable.getDoubleValue(col, row));
                    break;
                case DataType::BOOL:
                    rowData[col] = m_inputTable.getBoolValue(col, row) ? "true" : "false";
                    break;

                default:
                    rowData[col] = ""; // Default for unsupported types
                }
            }

            // Evaluate condition on this row
            if (m_condition.evaluate(rowData, columnNameToIndex))
            {
                // Add the row to result table if condition is satisfied
                for (size_t col = 0; col < colCount; ++col)
                {
                    const auto &column = m_inputTable.getColumns()[col];
                    switch (column.getType())
                    {
                    case DataType::INT:
                        resultTable.appendIntValue(col, m_inputTable.getIntValue(col, row));
                        break;
                    case DataType::FLOAT:
                        resultTable.appendFloatValue(col, m_inputTable.getFloatValue(col, row));
                        break;
                    case DataType::STRING:
                        resultTable.appendStringValue(col, m_inputTable.getStringValue(col, row));
                        break;
                    case DataType::VARCHAR:
                        resultTable.appendStringValue(col, m_inputTable.getStringValue(col, row));
                        break;
                    case DataType::DOUBLE:
                        resultTable.appendDoubleValue(col, m_inputTable.getDoubleValue(col, row));
                        break;
                    case DataType::BOOL:
                        resultTable.appendBoolValue(col, m_inputTable.getBoolValue(col, row));
                        break;
                    default:
                        // Handle default case or ignore
                        break;
                    }
                }

                // std::cout << resultTable.getColumnData(0).size() << std::endl;
                // std::cout << resultTable.getColumnData(1).size() << std::endl;
                // std::cout << resultTable.getColumnData(2).size() << std::endl;
                // std::cout << resultTable.getColumnData(3).size() << std::endl;
                // std::cout << resultTable.getColumnData(4).size() << std::endl;
                resultTable.finalizeRow();
            }
        }

        return resultTable;
    }

    // Project Implementation
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