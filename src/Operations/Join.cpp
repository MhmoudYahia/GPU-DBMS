#include "../../include/Operations/Join.hpp"
#include "../../src/Operations/JoinGPU.cu"
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <string>
#include <unordered_set>
#include <stdexcept>

namespace GPUDBMS
{
    Join::Join(const Table &leftTable, const Table &rightTable,
               const Condition &condition, JoinType joinType)
        : m_leftTable(leftTable), m_rightTable(rightTable),
          m_condition(condition), m_joinType(joinType)
    {
    }

    std::vector<Column> Join::createResultSchema() const
    {
        std::vector<Column> resultColumns;
        std::unordered_set<std::string> columnNames;

        // Add all columns from left table
        for (const auto &column : m_leftTable.getColumns())
        {
            resultColumns.push_back(column);
            columnNames.insert(column.getName());
        }

        // Add all columns from right table with prefix if there's a name conflict
        for (const auto &column : m_rightTable.getColumns())
        {
            std::string columnName = column.getName();

            // If column name already exists, prefix it with "right_"
            if (columnNames.find(columnName) != columnNames.end())
            {
                columnName = "right_" + columnName;
            }

            resultColumns.push_back(Column(columnName, column.getType()));
            columnNames.insert(columnName);
        }

        return resultColumns;
    }

    Table Join::execute(bool useGPU)
    {
        if (useGPU)
            return executeGPU();
        else
            return executeCPU();
    }

    Table Join::executeCPU()
    {
        // Create result table schema
        std::vector<Column> resultColumns = createResultSchema();
        Table resultTable(resultColumns);

        // Maps to track column names across tables
        std::unordered_map<std::string, int> leftColumnMap;
        std::unordered_map<std::string, DataType> leftColumnTypes;

        for (size_t i = 0; i < m_leftTable.getColumnCount(); ++i)
        {
            const auto &column = m_leftTable.getColumns()[i];
            leftColumnMap[column.getName()] = i;
            leftColumnTypes[column.getName()] = column.getType();
        }

        std::unordered_map<std::string, int> rightColumnMap;
        std::unordered_map<std::string, DataType> rightColumnTypes;

        for (size_t i = 0; i < m_rightTable.getColumnCount(); ++i)
        {
            const auto &column = m_rightTable.getColumns()[i];
            rightColumnMap[column.getName()] = i;
            rightColumnTypes[column.getName()] = column.getType();
        }

        // For each row in the left table
        for (size_t leftRow = 0; leftRow < m_leftTable.getRowCount(); ++leftRow)
        {
            bool leftRowMatched = false;

            // For each row in the right table
            for (size_t rightRow = 0; rightRow < m_rightTable.getRowCount(); ++rightRow)
            {
                // Build the combined row data for condition evaluation
                std::vector<std::string> combinedRowData;
                std::vector<DataType> combinedColTypes;
                std::unordered_map<std::string, int> columnNameToIndex;

                // First, prepare all column indices
                int colIndex = 0;
                
                // Add left table columns to the index map
                for (size_t col = 0; col < m_leftTable.getColumnCount(); ++col)
                {
                    const auto &column = m_leftTable.getColumns()[col];
                    columnNameToIndex[column.getName()] = colIndex++;
                }
                
                // Store the starting index for right table columns
                int rightStartIndex = colIndex;
                
                // Add right table columns to the index map
                for (size_t col = 0; col < m_rightTable.getColumnCount(); ++col)
                {
                    const auto &column = m_rightTable.getColumns()[col];
                    // Add both the original name and the prefixed name
                    std::string originalKey = column.getName();
                    std::string prefixedKey = "right_" + originalKey;
                    
                    columnNameToIndex[originalKey] = colIndex; // For the condition evaluation
                    columnNameToIndex[prefixedKey] = colIndex; // For result table access
                    
                    colIndex++;
                }
                
                // Now reset the column index to build the actual row data
                colIndex = 0;
                
                // Add data from left table
                for (size_t col = 0; col < m_leftTable.getColumnCount(); ++col)
                {
                    const auto &column = m_leftTable.getColumns()[col];
                    
                    std::string cellValue;
                    try
                    {
                        switch (column.getType())
                        {
                        case DataType::INT:
                            cellValue = std::to_string(m_leftTable.getIntValue(col, leftRow));
                            break;
                        case DataType::FLOAT:
                            cellValue = std::to_string(m_leftTable.getFloatValue(col, leftRow));
                            break;
                        case DataType::DOUBLE:
                            cellValue = std::to_string(m_leftTable.getDoubleValue(col, leftRow));
                            break;
                        case DataType::VARCHAR:
                        case DataType::STRING:
                            cellValue = m_leftTable.getStringValue(col, leftRow);
                            break;
                        case DataType::BOOL:
                            cellValue = m_leftTable.getBoolValue(col, leftRow) ? "true" : "false";
                            break;
                        }
                    }
                    catch (const std::exception &e)
                    {
                        std::cerr << "Error getting value from left table column " << column.getName() << ": " << e.what() << std::endl;
                        cellValue = "0"; // Default to safe value
                    }

                    combinedRowData.push_back(cellValue);
                    combinedColTypes.push_back(column.getType());
                    colIndex++;
                }

                // Add data from right table
                for (size_t col = 0; col < m_rightTable.getColumnCount(); ++col)
                {
                    const auto &column = m_rightTable.getColumns()[col];
                    
                    std::string cellValue;
                    try
                    {
                        switch (column.getType())
                        {
                        case DataType::INT:
                            cellValue = std::to_string(m_rightTable.getIntValue(col, rightRow));
                            break;
                        case DataType::FLOAT:
                            cellValue = std::to_string(m_rightTable.getFloatValue(col, rightRow));
                            break;
                        case DataType::DOUBLE:
                            cellValue = std::to_string(m_rightTable.getDoubleValue(col, rightRow));
                            break;
                        case DataType::VARCHAR:
                        case DataType::STRING:
                            cellValue = m_rightTable.getStringValue(col, rightRow);
                            break;
                        case DataType::BOOL:
                            cellValue = m_rightTable.getBoolValue(col, rightRow) ? "true" : "false";
                            break;
                        }
                    }
                    catch (const std::exception &e)
                    {
                        std::cerr << "Error getting value from right table column " << column.getName() << ": " << e.what() << std::endl;
                        cellValue = "0"; // Default to safe value
                    }

                    combinedRowData.push_back(cellValue);
                    combinedColTypes.push_back(column.getType());
                    colIndex++;
                }

                // Debug output after all data is prepared
                std::cout << "DEBUG: Left row " << leftRow << ", Right row " << rightRow << std::endl;
                std::cout << "DEBUG: Column name to index map: " << std::endl;
                for (const auto &entry : columnNameToIndex)
                {
                    std::cout << "  " << entry.first << " -> " << entry.second << std::endl;
                }
                
                // Evaluate join condition
                bool matchResult = false;
                try
                {
                    matchResult = m_condition.evaluate(combinedColTypes, combinedRowData, columnNameToIndex);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error evaluating join condition: " << e.what() << std::endl;
                    // Continue to next row pair
                    continue;
                }

                if (matchResult)
                {
                    leftRowMatched = true;

                    // Add joined row to result table
                    // First add columns from left table
                    for (size_t col = 0; col < m_leftTable.getColumnCount(); ++col)
                    {
                        const auto &column = m_leftTable.getColumns()[col];
                        switch (column.getType())
                        {
                        case DataType::INT:
                            resultTable.appendIntValue(col, m_leftTable.getIntValue(col, leftRow));
                            break;
                        case DataType::FLOAT:
                            resultTable.appendFloatValue(col, m_leftTable.getFloatValue(col, leftRow));
                            break;
                        case DataType::DOUBLE:
                            resultTable.appendDoubleValue(col, m_leftTable.getDoubleValue(col, leftRow));
                            break;
                        case DataType::VARCHAR:
                        case DataType::STRING:
                            resultTable.appendStringValue(col, m_leftTable.getStringValue(col, leftRow));
                            break;
                        case DataType::BOOL:
                            resultTable.appendBoolValue(col, m_leftTable.getBoolValue(col, leftRow));
                            break;
                        }
                    }

                    // Add columns from right table (with renamed columns to avoid duplicates)
                    size_t resultColIndex = m_leftTable.getColumnCount();
                    for (size_t col = 0; col < m_rightTable.getColumnCount(); ++col)
                    {
                        const auto &column = m_rightTable.getColumns()[col];

                        switch (column.getType())
                        {
                        case DataType::INT:
                            resultTable.appendIntValue(resultColIndex, m_rightTable.getIntValue(col, rightRow));
                            break;
                        case DataType::FLOAT:
                            resultTable.appendFloatValue(resultColIndex, m_rightTable.getFloatValue(col, rightRow));
                            break;
                        case DataType::DOUBLE:
                            resultTable.appendDoubleValue(resultColIndex, m_rightTable.getDoubleValue(col, rightRow));
                            break;
                        case DataType::VARCHAR:
                        case DataType::STRING:
                            resultTable.appendStringValue(resultColIndex, m_rightTable.getStringValue(col, rightRow));
                            break;
                        case DataType::BOOL:
                            resultTable.appendBoolValue(resultColIndex, m_rightTable.getBoolValue(col, rightRow));
                            break;
                        }

                        resultColIndex++;
                    }

                    resultTable.finalizeRow();
                }
            }

            // Handle LEFT JOIN unmatched rows
            if (!leftRowMatched && (m_joinType == JoinType::LEFT || m_joinType == JoinType::FULL))
            {
                // Add left row with NULL values for right columns
                for (size_t col = 0; col < m_leftTable.getColumnCount(); ++col)
                {
                    const auto &column = m_leftTable.getColumns()[col];
                    switch (column.getType())
                    {
                    case DataType::INT:
                        resultTable.appendIntValue(col, m_leftTable.getIntValue(col, leftRow));
                        break;
                    case DataType::FLOAT:
                        resultTable.appendFloatValue(col, m_leftTable.getFloatValue(col, leftRow));
                        break;
                    case DataType::DOUBLE:
                        resultTable.appendDoubleValue(col, m_leftTable.getDoubleValue(col, leftRow));
                        break;
                    case DataType::VARCHAR:
                    case DataType::STRING:
                        resultTable.appendStringValue(col, m_leftTable.getStringValue(col, leftRow));
                        break;
                    case DataType::BOOL:
                        resultTable.appendBoolValue(col, m_leftTable.getBoolValue(col, leftRow));
                        break;
                    }
                }

                // Add NULL values for right table columns
                for (size_t col = 0; col < m_rightTable.getColumnCount(); ++col)
                {
                    const auto &column = m_rightTable.getColumns()[col];
                    const size_t resultColIndex = m_leftTable.getColumnCount() + col;

                    switch (column.getType())
                    {
                    case DataType::INT:
                        resultTable.appendIntValue(resultColIndex, 0); // Default value for NULL
                        break;
                    case DataType::FLOAT:
                        resultTable.appendFloatValue(resultColIndex, 0.0f);
                        break;
                    case DataType::DOUBLE:
                        resultTable.appendDoubleValue(resultColIndex, 0.0);
                        break;
                    case DataType::VARCHAR:
                    case DataType::STRING:
                        resultTable.appendStringValue(resultColIndex, "");
                        break;
                    case DataType::BOOL:
                        resultTable.appendBoolValue(resultColIndex, false);
                        break;
                    }
                }

                resultTable.finalizeRow();
            }
        }

        return resultTable;
    }

    Table Join::executeGPU(){
            // Create result table schema
            std::vector<Column> resultColumns = createResultSchema();
            Table resultTable(resultColumns);
        
            // Maps to track column names across tables
            std::unordered_map<std::string, int> leftColumnMap;
            std::unordered_map<std::string, DataType> leftColumnTypes;

            for (size_t i = 0; i < m_leftTable.getColumnCount(); ++i)
            {
                const auto &column = m_leftTable.getColumns()[i];
                leftColumnMap[column.getName()] = i;
                leftColumnTypes[column.getName()] = column.getType();
            }

            std::unordered_map<std::string, int> rightColumnMap;
            std::unordered_map<std::string, DataType> rightColumnTypes;

            for (size_t i = 0; i < m_rightTable.getColumnCount(); ++i)
            {
                const auto &column = m_rightTable.getColumns()[i];
                rightColumnMap[column.getName()] = i;
                rightColumnTypes[column.getName()] = column.getType();
            }

            int leftRows = m_leftTable.getRowCount();
            int rightRows = m_rightTable.getRowCount();
            int leftCols = m_leftTable.getColumnCount();
            int rightCols = m_rightTable.getColumnCount();
        
            if(leftColumnTypes[m_condition.m_columnName]!=rightColumnTypes[m_condition.m_value])
                std::cerr << "Error the columns types are incompatible\n";
            
            switch (leftColumnTypes[m_condition.m_columnName])
            {
            case DataType::INT:
                launchJoinKernel<int>(&resultTable,leftCols,leftRows, rightCols, rightRows, DataType::INT);
                break;
            case DataType::FLOAT:
                launchJoinKernel<float>(&resultTable,leftCols,leftRows, rightCols, rightRows, DataType::FLOAT);
                break;
            case DataType::DOUBLE:
                launchJoinKernel<double>(&resultTable,leftCols,leftRows, rightCols, rightRows, DataType::DOUBLE);
                break;
            case DataType::VARCHAR:
            case DataType::STRING:
                launchJoinKernel<std::string>(&resultTable,leftCols,leftRows, rightCols, rightRows, DataType::STRING);
                break;
            case DataType::BOOL:
                launchJoinKernel<bool>(&resultTable,leftCols,leftRows, rightCols, rightRows, DataType::BOOL);
                break;
            }

            return resultTable;        
    }
 
} // namespace GPUDBMS