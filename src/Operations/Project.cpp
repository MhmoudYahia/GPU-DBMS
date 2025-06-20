#include "../../include/Operations/Project.hpp"
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
                throw std::runtime_error("Project -> Column not found: " + columnName);
            }
            m_columnIndices.push_back(index);
        }
    }

    Table Project::execute(bool useGPU)
    {
        if (useGPU)
            return executeGPU();
        else
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

        // For each row in the input table
        for (size_t row = 0; row < rowCount; ++row)
        {
            // For each column in the projection
            for (size_t resultCol = 0; resultCol < m_columnIndices.size(); ++resultCol)
            {
                int sourceCol = m_columnIndices[resultCol];
                const auto &column = m_inputTable.getColumns()[sourceCol];

                // Copy the value based on type
                switch (column.getType())
                {
                case DataType::INT:
                    resultTable.appendIntValue(resultCol, m_inputTable.getIntValue(sourceCol, row));
                    break;
                case DataType::FLOAT:
                    resultTable.appendFloatValue(resultCol, m_inputTable.getFloatValue(sourceCol, row));
                    break;
                case DataType::DOUBLE:
                    resultTable.appendDoubleValue(resultCol, m_inputTable.getDoubleValue(sourceCol, row));
                    break;
                case DataType::VARCHAR:
                case DataType::STRING:
                    resultTable.appendStringValue(resultCol, m_inputTable.getStringValue(sourceCol, row));
                    break;
                case DataType::BOOL:
                    resultTable.appendBoolValue(resultCol, m_inputTable.getBoolValue(sourceCol, row));  
                    break;
                
                case DataType::DATE:
                case DataType::DATETIME:
                    try {
                        std::string dateValue = m_inputTable.getDateTimeValue(sourceCol, row);
                        resultTable.appendStringValue(resultCol, dateValue);
                    } catch (const std::exception& e) {
                        // Fallback to string if datetime-specific method fails
                        std::cout << "Warning: Falling back to string for datetime column: " << e.what() << std::endl;
                        resultTable.appendStringValue(resultCol, m_inputTable.getStringValue(sourceCol, row));
                    }
                    break;
                default:
                    throw std::runtime_error("Unsupported data type for projection: " + column.getName());
                }
            }

            // Finalize the row after copying all column values
            resultTable.finalizeRow();
        }

        return resultTable;
    }

    Table Project::executeGPU()
    {
        std::vector<std::string> projectColumns;

        for (int index : m_columnIndices)
        {
            projectColumns.push_back(m_inputTable.getColumns()[index].getName());
        }

        return launchProjectKernel(m_inputTable, projectColumns);
    }

} // namespace GPUDBMS