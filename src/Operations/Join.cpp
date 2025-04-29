#include "../../include/Operations/Join.hpp"
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_set>

namespace GPUDBMS
{
    Join::Join(const Table &leftTable, const Table &rightTable,
               const Condition &condition, JoinType joinType)
        : m_leftTable(leftTable), m_rightTable(rightTable),
          m_condition(condition), m_joinType(joinType)
    {
    }

    Table Join::execute()
    {
        // For now, just call CPU implementation
        // TODO: Implement GPU version
        return executeCPU();
    }

    std::vector<Column> Join::createResultSchema() const
    {
        std::vector<Column> resultColumns;

        // Add all columns from left table
        for (const auto &column : m_leftTable.getColumns())
        {
            resultColumns.push_back(column);
        }

        // Add columns from right table, handling name conflicts
        for (const auto &column : m_rightTable.getColumns())
        {
            std::string colName = column.getName();

            // Check for conflicts with left table columns
            bool hasConflict = false;
            for (const auto &leftCol : m_leftTable.getColumns())
            {
                if (leftCol.getName() == colName)
                {
                    hasConflict = true;
                    break;
                }
            }

            // If there's a name conflict, prefix with right_
            if (hasConflict)
            {
                resultColumns.push_back(Column("right_" + colName, column.getType()));
            }
            else
            {
                resultColumns.push_back(column);
            }
        }

        return resultColumns;
    }

    Table Join::executeCPU()
    {
        // Create result table schema
        std::vector<Column> resultColumns = createResultSchema();
        Table resultTable(resultColumns);

        // Prepare column name to index maps for both tables
        std::unordered_map<std::string, int> leftColumnMap;
        std::unordered_map<std::string, int> rightColumnMap;

        for (size_t i = 0; i < m_leftTable.getColumnCount(); ++i)
        {
            leftColumnMap[m_leftTable.getColumns()[i].getName()] = static_cast<int>(i);
        }

        for (size_t i = 0; i < m_rightTable.getColumnCount(); ++i)
        {
            rightColumnMap[m_rightTable.getColumns()[i].getName()] = static_cast<int>(i);
        }

        // Track which column names have conflicts (need right_ prefix)
        std::unordered_set<std::string> conflictingNames;

        for (const auto &leftCol : m_leftTable.getColumns())
        {
            for (const auto &rightCol : m_rightTable.getColumns())
            {
                if (leftCol.getName() == rightCol.getName())
                {
                    conflictingNames.insert(leftCol.getName());
                    break;
                }
            }
        }

        std::vector<DataType> leftColTypes = m_leftTable.getColumnsType();
        std::vector<DataType> rightColTypes = m_rightTable.getColumnsType();

        // Debug: Print column info
        std::cout << "Left table columns:" << std::endl;
        for (const auto &col : m_leftTable.getColumns())
        {
            std::cout << "  " << col.getName() << std::endl;
        }

        std::cout << "Right table columns:" << std::endl;
        for (const auto &col : m_rightTable.getColumns())
        {
            std::cout << "  " << col.getName() << std::endl;
        }

        std::cout << "Conflicting column names:" << std::endl;
        for (const auto &name : conflictingNames)
        {
            std::cout << "  " << name << std::endl;
        }

        // Process each row in the left table
        for (size_t leftRow = 0; leftRow < m_leftTable.getRowCount(); ++leftRow)
        {
            bool matchFound = false;

            // Process each row in the right table
            for (size_t rightRow = 0; rightRow < m_rightTable.getRowCount(); ++rightRow)
            {
                // Create combined row data for condition evaluation
                std::vector<std::string> combinedData;
                std::unordered_map<std::string, int> combinedColumnMap;

                // First, add all columns from left table
                for (size_t col = 0; col < m_leftTable.getColumnCount(); ++col)
                {
                    const auto &column = m_leftTable.getColumns()[col];
                    std::string colName = column.getName();
                    combinedColumnMap[colName] = combinedData.size();

                    // Convert value to string based on type
                    switch (column.getType())
                    {
                    case DataType::INT:
                        combinedData.push_back(std::to_string(m_leftTable.getIntValue(col, leftRow)));
                        break;
                    case DataType::FLOAT:
                        combinedData.push_back(std::to_string(m_leftTable.getFloatValue(col, leftRow)));
                        break;
                    case DataType::DOUBLE:
                        combinedData.push_back(std::to_string(m_leftTable.getDoubleValue(col, leftRow)));
                        break;
                    case DataType::VARCHAR:
                    case DataType::STRING:
                        combinedData.push_back(m_leftTable.getStringValue(col, leftRow));
                        break;
                    case DataType::BOOL:
                        combinedData.push_back(m_leftTable.getBoolValue(col, leftRow) ? "true" : "false");
                        break;
                    }
                }

                // Then add all columns from right table
                for (size_t col = 0; col < m_rightTable.getColumnCount(); ++col)
                {
                    const auto &column = m_rightTable.getColumns()[col];
                    std::string colName = column.getName();
                    std::string mappedName = colName;

                    // Check if this column name conflicts with a left table column
                    if (conflictingNames.find(colName) != conflictingNames.end())
                    {
                        mappedName = "right_" + colName;
                    }

                    combinedColumnMap[mappedName] = combinedData.size();

                    // Convert value to string based on type
                    switch (column.getType())
                    {
                    case DataType::INT:
                        combinedData.push_back(std::to_string(m_rightTable.getIntValue(col, rightRow)));
                        break;
                    case DataType::FLOAT:
                        combinedData.push_back(std::to_string(m_rightTable.getFloatValue(col, rightRow)));
                        break;
                    case DataType::DOUBLE:
                        combinedData.push_back(std::to_string(m_rightTable.getDoubleValue(col, rightRow)));
                        break;
                    case DataType::VARCHAR:
                    case DataType::STRING:
                        combinedData.push_back(m_rightTable.getStringValue(col, rightRow));
                        break;
                    case DataType::BOOL:
                        combinedData.push_back(m_rightTable.getBoolValue(col, rightRow) ? "true" : "false");
                        break;
                    }
                }

                // Debug: Print combined column map and data for first rows
                if (leftRow == 0 && rightRow == 0)
                {
                    std::cout << "Debug - Combined column map:" << std::endl;
                    for (const auto &[key, value] : combinedColumnMap)
                    {
                        std::cout << "  " << key << " -> " << value << " (" << combinedData[value] << ")" << std::endl;
                    }
                }

                // Try to evaluate the join condition
                try
                {
                    // Need to provide the column types as the third parameter
                    std::vector<DataType> combinedColTypes;
                    // Add types from left table
                    for (size_t i = 0; i < m_leftTable.getColumnCount(); ++i)
                    {
                        combinedColTypes.push_back(m_leftTable.getColumns()[i].getType());
                    }
                    // Add types from right table
                    for (size_t i = 0; i < m_rightTable.getColumnCount(); ++i)
                    {
                        combinedColTypes.push_back(m_rightTable.getColumns()[i].getType());
                    }

                    if (m_condition.evaluate(combinedColTypes, combinedData, combinedColumnMap))
                    {
                        matchFound = true;

                        // Add row to result table (first left columns, then right columns)
                        size_t resultCol = 0;

                        // Add values from left table
                        for (size_t col = 0; col < m_leftTable.getColumnCount(); ++col, ++resultCol)
                        {
                            const auto &column = m_leftTable.getColumns()[col];
                            switch (column.getType())
                            {
                            case DataType::INT:
                                resultTable.appendIntValue(resultCol, m_leftTable.getIntValue(col, leftRow));
                                break;
                            case DataType::FLOAT:
                                resultTable.appendFloatValue(resultCol, m_leftTable.getFloatValue(col, leftRow));
                                break;
                            case DataType::DOUBLE:
                                resultTable.appendDoubleValue(resultCol, m_leftTable.getDoubleValue(col, leftRow));
                                break;
                            case DataType::VARCHAR:
                            case DataType::STRING:
                                resultTable.appendStringValue(resultCol, m_leftTable.getStringValue(col, leftRow));
                                break;
                            case DataType::BOOL:
                                resultTable.appendBoolValue(resultCol, m_leftTable.getBoolValue(col, leftRow));
                                break;
                            }
                        }

                        // Add values from right table
                        for (size_t col = 0; col < m_rightTable.getColumnCount(); ++col, ++resultCol)
                        {
                            const auto &column = m_rightTable.getColumns()[col];
                            switch (column.getType())
                            {
                            case DataType::INT:
                                resultTable.appendIntValue(resultCol, m_rightTable.getIntValue(col, rightRow));
                                break;
                            case DataType::FLOAT:
                                resultTable.appendFloatValue(resultCol, m_rightTable.getFloatValue(col, rightRow));
                                break;
                            case DataType::DOUBLE:
                                resultTable.appendDoubleValue(resultCol, m_rightTable.getDoubleValue(col, rightRow));
                                break;
                            case DataType::VARCHAR:
                            case DataType::STRING:
                                resultTable.appendStringValue(resultCol, m_rightTable.getStringValue(col, rightRow));
                                break;
                            case DataType::BOOL:
                                resultTable.appendBoolValue(resultCol, m_rightTable.getBoolValue(col, rightRow));
                                break;
                            }
                        }

                        resultTable.finalizeRow();
                    }
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error evaluating join condition: " << e.what() << std::endl;
                    std::cerr << "Left row: " << leftRow << ", Right row: " << rightRow << std::endl;
                    // Continue to next row pair
                }
            }

            // For LEFT and FULL joins, add the left row with NULL values for right table if no match found
            if (!matchFound && (m_joinType == JoinType::LEFT || m_joinType == JoinType::FULL))
            {
                size_t resultCol = 0;

                // Add values from left table
                for (size_t col = 0; col < m_leftTable.getColumnCount(); ++col, ++resultCol)
                {
                    const auto &column = m_leftTable.getColumns()[col];
                    switch (column.getType())
                    {
                    case DataType::INT:
                        resultTable.appendIntValue(resultCol, m_leftTable.getIntValue(col, leftRow));
                        break;
                    case DataType::FLOAT:
                        resultTable.appendFloatValue(resultCol, m_leftTable.getFloatValue(col, leftRow));
                        break;
                    case DataType::DOUBLE:
                        resultTable.appendDoubleValue(resultCol, m_leftTable.getDoubleValue(col, leftRow));
                        break;
                    case DataType::VARCHAR:
                    case DataType::STRING:
                        resultTable.appendStringValue(resultCol, m_leftTable.getStringValue(col, leftRow));
                        break;
                    case DataType::BOOL:
                        resultTable.appendBoolValue(resultCol, m_leftTable.getBoolValue(col, leftRow));
                        break;
                    }
                }

                // Add NULL values for right table
                for (size_t col = 0; col < m_rightTable.getColumnCount(); ++col, ++resultCol)
                {
                    const auto &column = m_rightTable.getColumns()[col];
                    switch (column.getType())
                    {
                    case DataType::INT:
                        resultTable.appendIntValue(resultCol, 0); // NULL for int
                        break;
                    case DataType::FLOAT:
                        resultTable.appendFloatValue(resultCol, 0.0f); // NULL for float
                        break;
                    case DataType::DOUBLE:
                        resultTable.appendDoubleValue(resultCol, 0.0); // NULL for double
                        break;
                    case DataType::VARCHAR:
                    case DataType::STRING:
                        resultTable.appendStringValue(resultCol, "NULL"); // NULL for string
                        break;
                    case DataType::BOOL:
                        resultTable.appendBoolValue(resultCol, false); // NULL for bool
                        break;
                    }
                }

                resultTable.finalizeRow();
            }
        }

        // For RIGHT and FULL joins, add right rows that don't match any left row
        if (m_joinType == JoinType::RIGHT || m_joinType == JoinType::FULL)
        {
            // Track right rows that have been matched
            std::vector<bool> rightRowMatched(m_rightTable.getRowCount(), false);

            // Find which right rows matched with any left row
            for (size_t leftRow = 0; leftRow < m_leftTable.getRowCount(); ++leftRow)
            {
                for (size_t rightRow = 0; rightRow < m_rightTable.getRowCount(); ++rightRow)
                {
                    // Create combined row data for condition evaluation
                    std::vector<std::string> combinedData;
                    std::unordered_map<std::string, int> combinedColumnMap;

                    // Add left and right data (similar to above)
                    // Add left table data
                    for (size_t col = 0; col < m_leftTable.getColumnCount(); ++col)
                    {
                        const auto &column = m_leftTable.getColumns()[col];
                        std::string colName = column.getName();
                        combinedColumnMap[colName] = combinedData.size();

                        // Convert value to string based on type
                        switch (column.getType())
                        {
                        case DataType::INT:
                            combinedData.push_back(std::to_string(m_leftTable.getIntValue(col, leftRow)));
                            break;
                        case DataType::FLOAT:
                            combinedData.push_back(std::to_string(m_leftTable.getFloatValue(col, leftRow)));
                            break;
                        case DataType::DOUBLE:
                            combinedData.push_back(std::to_string(m_leftTable.getDoubleValue(col, leftRow)));
                            break;
                        case DataType::VARCHAR:
                        case DataType::STRING:
                            combinedData.push_back(m_leftTable.getStringValue(col, leftRow));
                            break;
                        case DataType::BOOL:
                            combinedData.push_back(m_leftTable.getBoolValue(col, leftRow) ? "true" : "false");
                            break;
                        }
                    }

                    // Add right table data
                    for (size_t col = 0; col < m_rightTable.getColumnCount(); ++col)
                    {
                        const auto &column = m_rightTable.getColumns()[col];
                        std::string colName = column.getName();
                        std::string mappedName = colName;

                        // Handle column name conflicts
                        if (conflictingNames.find(colName) != conflictingNames.end())
                        {
                            mappedName = "right_" + colName;
                        }

                        combinedColumnMap[mappedName] = combinedData.size();

                        // Convert value to string based on type
                        switch (column.getType())
                        {
                        case DataType::INT:
                            combinedData.push_back(std::to_string(m_rightTable.getIntValue(col, rightRow)));
                            break;
                        case DataType::FLOAT:
                            combinedData.push_back(std::to_string(m_rightTable.getFloatValue(col, rightRow)));
                            break;
                            case DataType::DOUBLE:
                                combinedData.push_back(std::to_string(m_rightTable.getDoubleValue(col, rightRow)));
                                break;
                            case DataType::VARCHAR:
                            case DataType::STRING:
                                combinedData.push_back(m_rightTable.getStringValue(col, rightRow));
                                break;
                            case DataType::BOOL:
                                combinedData.push_back(m_rightTable.getBoolValue(col, rightRow) ? "true" : "false");
                                break;
                            }
                        }
    
                        // If the condition is true, mark the right row as matched
                        try
                        {
                            // Need to provide the column types as the third parameter
                            std::vector<DataType> combinedColTypes;
                            // Add types from left table
                            for (size_t i = 0; i < m_leftTable.getColumnCount(); ++i)
                            {
                                combinedColTypes.push_back(m_leftTable.getColumns()[i].getType());
                            }
                            // Add types from right table
                            for (size_t i = 0; i < m_rightTable.getColumnCount(); ++i)
                            {
                                combinedColTypes.push_back(m_rightTable.getColumns()[i].getType());
                            }
    
                            if (m_condition.evaluate(combinedColTypes, combinedData, combinedColumnMap))
                            {
                                rightRowMatched[rightRow] = true;
                            }
                        }
                        catch (const std::exception &)
                        {
                            // Ignore evaluation errors for matching
                        }
                    }
                }

                // Add right rows that weren't matched
                for (size_t rightRow = 0; rightRow < m_rightTable.getRowCount(); ++rightRow)
                {
                    if (!rightRowMatched[rightRow])
                    {
                        size_t resultCol = 0;

                        // Add NULL values for left table
                        for (size_t col = 0; col < m_leftTable.getColumnCount(); ++col, ++resultCol)
                        {
                            const auto &column = m_leftTable.getColumns()[col];
                            switch (column.getType())
                            {
                            case DataType::INT:
                                resultTable.appendIntValue(resultCol, 0); // NULL for int
                                break;
                            case DataType::FLOAT:
                                resultTable.appendFloatValue(resultCol, 0.0f); // NULL for float
                                break;
                            case DataType::DOUBLE:
                                resultTable.appendDoubleValue(resultCol, 0.0); // NULL for double
                                break;
                            case DataType::VARCHAR:
                            case DataType::STRING:
                                resultTable.appendStringValue(resultCol, "NULL"); // NULL for string
                                break;
                            case DataType::BOOL:
                                resultTable.appendBoolValue(resultCol, false); // NULL for bool
                                break;
                            }
                        }

                        // Add values from right table
                        for (size_t col = 0; col < m_rightTable.getColumnCount(); ++col, ++resultCol)
                        {
                            const auto &column = m_rightTable.getColumns()[col];
                            switch (column.getType())
                            {
                            case DataType::INT:
                                resultTable.appendIntValue(resultCol, m_rightTable.getIntValue(col, rightRow));
                                break;
                            case DataType::FLOAT:
                                resultTable.appendFloatValue(resultCol, m_rightTable.getFloatValue(col, rightRow));
                                break;
                            case DataType::DOUBLE:
                                resultTable.appendDoubleValue(resultCol, m_rightTable.getDoubleValue(col, rightRow));
                                break;
                            case DataType::VARCHAR:
                            case DataType::STRING:
                                resultTable.appendStringValue(resultCol, m_rightTable.getStringValue(col, rightRow));
                                break;
                            case DataType::BOOL:
                                resultTable.appendBoolValue(resultCol, m_rightTable.getBoolValue(col, rightRow));
                                break;
                            }
                        }

                        resultTable.finalizeRow();
                    }
                }
            }

            return resultTable;
        }

    } // namespace GPUDBMS