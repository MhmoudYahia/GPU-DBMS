#include "../../include/Operations/Filter.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
<<<<<<< Updated upstream

namespace GPUDBMS
{

    Filter::Filter(const Table &inputTable, const std::vector<std::unique_ptr<Condition>> &conditions, bool isAnd)
        : m_inputTable(inputTable), m_isAnd(isAnd)
    {
        // Clone each condition to store a deep copy
        for (const auto &cond : conditions)
        {
            m_conditions.push_back(cond->clone());
        }
=======
#include <unordered_map>

namespace GPUDBMS
{
    // Filter Implementation
    Filter::Filter(const Table &inputTable, const Condition &condition)
        : m_inputTable(inputTable), m_condition(condition)
    {
>>>>>>> Stashed changes
    }

    Table Filter::execute()
    {
        // For now, just call CPU implementation
        // TODO: Implement GPU version
        return executeCPU();
    }

    Table Filter::executeCPU()
    {
        Table resultTable = m_inputTable.createEmptyWithSameSchema();
        const size_t rowCount = m_inputTable.getRowCount();
        const size_t colCount = m_inputTable.getColumnCount();

<<<<<<< Updated upstream
=======
        // Create a column name to index mapping
>>>>>>> Stashed changes
        std::unordered_map<std::string, int> columnNameToIndex;
        for (size_t i = 0; i < colCount; ++i)
        {
            columnNameToIndex[m_inputTable.getColumns()[i].getName()] = static_cast<int>(i);
        }

<<<<<<< Updated upstream
        std::vector<DataType> colsType = m_inputTable.getColumnsType();

        // For each row in the input table
        for (size_t row = 0; row < rowCount; ++row)
        {
            // Extract current row data
=======
        // For each row in the input table
        for (size_t row = 0; row < rowCount; ++row)
        {
            // Extract current row data for condition evaluation
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
                    rowData[col] = m_inputTable.getStringValue(col, row);
                    break;
=======
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
            // Evaluate all conditions on this row
            bool rowSatisfiesConditions;

            if (m_isAnd)
            {
                // For AND logic, all conditions must be satisfied
                rowSatisfiesConditions = true;
                for (const auto &condition : m_conditions)
                {
                    if (!condition->evaluate(colsType, rowData, columnNameToIndex))
                    {
                        rowSatisfiesConditions = false;
                        break;
                    }
                }
            }
            else
            {
                // For OR logic, at least one condition must be satisfied
                rowSatisfiesConditions = false;
                for (const auto &condition : m_conditions)
                {
                    if (condition->evaluate(colsType, rowData, columnNameToIndex))
                    {
                        rowSatisfiesConditions = true;
                        break;
                    }
                }
            }

            // Add the row to result table if it satisfies the combined conditions
            if (rowSatisfiesConditions)
            {
=======
            // Evaluate condition on this row
            if (m_condition.evaluate(rowData, columnNameToIndex))
            {
                // Add the row to result table if condition is satisfied
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
                        resultTable.appendStringValue(col, m_inputTable.getStringValue(col, row));
                        break;
=======
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
                        // Handle default case or ignore
=======
                        // Ignore unsupported types
>>>>>>> Stashed changes
                        break;
                    }
                }
                resultTable.finalizeRow();
            }
        }

        return resultTable;
    }

} // namespace GPUDBMS