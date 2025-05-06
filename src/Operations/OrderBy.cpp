#include "../../include/Operations/OrderBy.hpp"
#include <algorithm>
#include <iostream>

namespace GPUDBMS
{
    OrderBy::OrderBy(const Table &inputTable,
                     const std::string &sortColumn,
                     SortOrder order)
        : m_inputTable(inputTable)
    {
        m_sortColumns.push_back(sortColumn);
        m_sortOrders.push_back(order);
        resolveColumnIndices();
    }

    OrderBy::OrderBy(const Table &inputTable,
                     const std::vector<std::string> &sortColumns,
                     const std::vector<SortOrder> &sortOrders)
        : m_inputTable(inputTable), m_sortColumns(sortColumns)
    {
        if (sortOrders.size() == sortColumns.size())
        {
            m_sortOrders = sortOrders;
        }
        else if (sortOrders.empty())
        {
            // Default to ascending order if no orders specified
            m_sortOrders.resize(sortColumns.size(), SortOrder::ASC);
        }
        else
        {
            throw std::runtime_error("Sort columns and sort orders must have the same size");
        }

        resolveColumnIndices();
    }

    void OrderBy::resolveColumnIndices()
    {
        m_columnIndices.clear();
        for (const auto &columnName : m_sortColumns)
        {
            int index = m_inputTable.getColumnIndex(columnName);
            if (index < 0)
            {
                throw std::runtime_error("Column not found: " + columnName);
            }
            m_columnIndices.push_back(index);
        }
    }

    Table OrderBy::execute(bool useGPU)
    {
        if (useGPU)
        {
            return executeGPU();
        }
        else
            return executeCPU();
    }

    bool OrderBy::compareRows(size_t rowIndexA, size_t rowIndexB) const
    {
        // Compare rows based on sort columns and orders
        for (size_t i = 0; i < m_columnIndices.size(); ++i)
        {
            int columnIndex = m_columnIndices[i];
            const auto &column = m_inputTable.getColumns()[columnIndex];
            bool isAscending = (m_sortOrders[i] == SortOrder::ASC);

            // Compare based on column type
            int comparison = 0;
            switch (column.getType())
            {
            case DataType::INT:
            {
                int valueA = m_inputTable.getIntValue(columnIndex, rowIndexA);
                int valueB = m_inputTable.getIntValue(columnIndex, rowIndexB);
                comparison = (valueA < valueB) ? -1 : (valueA > valueB) ? 1
                                                                        : 0;
                break;
            }
            case DataType::FLOAT:
            {
                float valueA = m_inputTable.getFloatValue(columnIndex, rowIndexA);
                float valueB = m_inputTable.getFloatValue(columnIndex, rowIndexB);
                comparison = (valueA < valueB) ? -1 : (valueA > valueB) ? 1
                                                                        : 0;
                break;
            }
            case DataType::DOUBLE:
            {
                double valueA = m_inputTable.getDoubleValue(columnIndex, rowIndexA);
                double valueB = m_inputTable.getDoubleValue(columnIndex, rowIndexB);
                comparison = (valueA < valueB) ? -1 : (valueA > valueB) ? 1
                                                                        : 0;
                break;
            }
            case DataType::VARCHAR:
            case DataType::STRING:
            {
                std::string valueA = m_inputTable.getStringValue(columnIndex, rowIndexA);
                std::string valueB = m_inputTable.getStringValue(columnIndex, rowIndexB);
                comparison = valueA.compare(valueB);
                break;
            }
            case DataType::BOOL:
            {
                bool valueA = m_inputTable.getBoolValue(columnIndex, rowIndexA);
                bool valueB = m_inputTable.getBoolValue(columnIndex, rowIndexB);
                comparison = (!valueA && valueB) ? -1 : (valueA && !valueB) ? 1
                                                                            : 0;
                break;
            }
            }

            // Apply sort order
            if (comparison != 0)
            {
                return isAscending ? (comparison < 0) : (comparison > 0);
            }
        }

        // If all comparisons are equal, maintain original order (stable sort)
        return false;
    }

    Table OrderBy::executeCPU()
    {
        // Create a copy of the input table
        Table resultTable = m_inputTable.createEmptyWithSameSchema();

        // No need to sort if the table is empty
        if (m_inputTable.getRowCount() == 0)
        {
            return resultTable;
        }

        // Create a vector of row indices for sorting
        std::vector<size_t> rowIndices(m_inputTable.getRowCount());
        for (size_t i = 0; i < rowIndices.size(); ++i)
        {
            rowIndices[i] = i;
        }

        // Sort row indices based on the sort columns and orders
        std::sort(rowIndices.begin(), rowIndices.end(),
                  [this](size_t a, size_t b)
                  { return compareRows(a, b); });

        // Copy rows in sorted order
        const size_t colCount = m_inputTable.getColumnCount();
        for (size_t rowIdx : rowIndices)
        {
            // Copy each column's value for this row
            for (size_t col = 0; col < colCount; ++col)
            {
                const auto &column = m_inputTable.getColumns()[col];
                switch (column.getType())
                {
                case DataType::INT:
                    resultTable.appendIntValue(col, m_inputTable.getIntValue(col, rowIdx));
                    break;
                case DataType::FLOAT:
                    resultTable.appendFloatValue(col, m_inputTable.getFloatValue(col, rowIdx));
                    break;
                case DataType::DOUBLE:
                    resultTable.appendDoubleValue(col, m_inputTable.getDoubleValue(col, rowIdx));
                    break;
                case DataType::VARCHAR:
                case DataType::STRING:
                    resultTable.appendStringValue(col, m_inputTable.getStringValue(col, rowIdx));
                    break;
                case DataType::BOOL:
                    resultTable.appendBoolValue(col, m_inputTable.getBoolValue(col, rowIdx));
                    break;
                }
            }
            resultTable.finalizeRow();
        }

        return resultTable;
    }

    Table OrderBy::executeGPU()
    {
        return launchOrderByKernel(m_inputTable, m_sortColumns, m_sortOrders);
    }

} // namespace GPUDBMS