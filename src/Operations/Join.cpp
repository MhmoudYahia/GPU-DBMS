#include "../../include/Operations/Join.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>

namespace GPUDBMS
{

    Join::Join(const Table &leftTable, const Table &rightTable, const Condition &condition)
        : m_leftTable(leftTable), m_rightTable(rightTable), m_condition(condition)
    {
    }

    Table Join::execute(bool useGPU)
    {
        if (useGPU)
            return executeGPU();
        else
            return executeCPU();
    }

    std::vector<Column> Join::createJoinedSchema() const
    {
        std::vector<Column> joinedColumns;
        std::unordered_set<std::string> columnNames;

        // Add all columns from left table
        for (const auto &col : m_leftTable.getColumns())
        {
            joinedColumns.push_back(col);
            columnNames.insert(col.getName());
        }

        // Add all columns from right table with prefixes for duplicates
        for (const auto &col : m_rightTable.getColumns())
        {
            std::string colName = col.getName();

            // If the column name already exists, prefix it with "right_"
            if (columnNames.find(colName) != columnNames.end())
            {
                Column prefixedCol = col;
                prefixedCol.setName("right_" + colName);
                joinedColumns.push_back(prefixedCol);
            }
            else
            {
                joinedColumns.push_back(col);
                columnNames.insert(colName);
            }
        }

        return joinedColumns;
    }

    Table Join::executeCPU()
    {
        // Create the schema for the joined table
        std::vector<Column> joinedColumns = createJoinedSchema();
        Table resultTable(joinedColumns);

        // Get row counts
        const size_t leftRowCount = m_leftTable.getRowCount();
        const size_t rightRowCount = m_rightTable.getRowCount();

        // Get column counts for both tables
        const size_t leftColCount = m_leftTable.getColumnCount();
        const size_t rightColCount = m_rightTable.getColumnCount();
        const size_t totalColCount = leftColCount + rightColCount;

        // Create column name to index mappings
        std::unordered_map<std::string, int> columnNameToIndex = resultTable.getColumnNameToIndex();

        // Get column types for condition evaluation
        std::vector<DataType> colTypes;

        std::vector<DataType> colTypesL = m_leftTable.getColumnsType();
        std::vector<DataType> colTypesR = m_rightTable.getColumnsType();
        colTypes.insert(colTypes.end(), colTypesL.begin(), colTypesL.end());
        colTypes.insert(colTypes.end(), colTypesR.begin(), colTypesR.end());

        // // Add left table column types
        // for (const auto &col : m_leftTable.getColumns())
        // {
        //     colTypes.push_back(col.getType());
        // }

        // // Add right table column types
        // for (const auto &col : m_rightTable.getColumns())
        // {
        //     colTypes.push_back(col.getType());
        // }

        int matchCount = 0;

        // Perform a nested loop join (can be optimized for hash join or sort-merge join)
        for (size_t leftRow = 0; leftRow < leftRowCount; ++leftRow)
        {
            for (size_t rightRow = 0; rightRow < rightRowCount; ++rightRow)
            {
                // Combine row data from both tables
                std::vector<std::string> combinedRowData(totalColCount);
                bool rowHasError = false;

                // Extract data from left row
                for (size_t col = 0; col < leftColCount; ++col)
                {
                    try
                    {
                        const auto &column = m_leftTable.getColumns()[col];
                        switch (column.getType())
                        {
                        case DataType::INT:
                            combinedRowData[col] = std::to_string(m_leftTable.getIntValue(col, leftRow));
                            break;
                        case DataType::FLOAT:
                            combinedRowData[col] = std::to_string(m_leftTable.getFloatValue(col, leftRow));
                            break;
                        case DataType::STRING:
                        case DataType::VARCHAR:
                            combinedRowData[col] = m_leftTable.getStringValue(col, leftRow);
                            break;
                        case DataType::DOUBLE:
                            combinedRowData[col] = std::to_string(m_leftTable.getDoubleValue(col, leftRow));
                            break;
                        case DataType::BOOL:
                            combinedRowData[col] = m_leftTable.getBoolValue(col, leftRow) ? "true" : "false";
                            break;
                        case DataType::DATE:
                        case DataType::DATETIME:
                            combinedRowData[col] = m_leftTable.getDateTimeValue(col, leftRow);
                            break;
                        default:
                            combinedRowData[col] = ""; // Default for unsupported types
                        }
                    }
                    catch (const std::exception &e)
                    {
                        std::cerr << "Error getting left table value for row " << leftRow
                                  << ", column " << col << ": " << e.what() << std::endl;
                        combinedRowData[col] = "";
                        rowHasError = true;
                    }
                }

                // Extract data from right row
                for (size_t col = 0; col < rightColCount; ++col)
                {
                    size_t combinedCol = col + leftColCount;
                    try
                    {
                        const auto &column = m_rightTable.getColumns()[col];
                        switch (column.getType())
                        {
                        case DataType::INT:
                            combinedRowData[combinedCol] = std::to_string(m_rightTable.getIntValue(col, rightRow));
                            break;
                        case DataType::FLOAT:
                            combinedRowData[combinedCol] = std::to_string(m_rightTable.getFloatValue(col, rightRow));
                            break;
                        case DataType::STRING:
                        case DataType::VARCHAR:
                            combinedRowData[combinedCol] = m_rightTable.getStringValue(col, rightRow);
                            break;
                        case DataType::DOUBLE:
                            combinedRowData[combinedCol] = std::to_string(m_rightTable.getDoubleValue(col, rightRow));
                            break;
                        case DataType::BOOL:
                            combinedRowData[combinedCol] = m_rightTable.getBoolValue(col, rightRow) ? "true" : "false";
                            break;
                        case DataType::DATE:
                        case DataType::DATETIME:
                            combinedRowData[combinedCol] = m_rightTable.getDateTimeValue(col, rightRow);
                            break;
                        default:
                            combinedRowData[combinedCol] = ""; // Default for unsupported types
                        }
                    }
                    catch (const std::exception &e)
                    {
                        std::cerr << "Error getting right table value for row " << rightRow
                                  << ", column " << col << ": " << e.what() << std::endl;
                        combinedRowData[combinedCol] = "";
                        rowHasError = true;
                    }
                }

                if (rowHasError)
                {
                    continue; // Skip rows with errors
                }

                // Evaluate join condition
                bool matches = false;

                // for(auto & col : colTypes)
                // {
                //     switch (col)
                //     {
                //     case DataType::INT:
                //         printf("col type: INT\n");
                //         break;
                //     case DataType::FLOAT:
                //         printf("col type: FLOAT\n");
                //         break;
                //     case DataType::STRING:
                //         printf("col type: STRING\n");
                //         break;
                //     case DataType::DOUBLE:
                //         printf("col type: DOUBLE\n");
                //         break;
                //     case DataType::BOOL:    
                //         printf("col type: BOOL\n");
                //         break;
                //     case DataType::DATE:
                        
                //         printf("col type: DATE\n");
                //         break;      
                //     case DataType::DATETIME:
                //         printf("col type: DATETIME\n");
                //         break;
                //     default:
                //         printf("col type: UNKNOWN\n");
                //         break;
                //     }
                // }
                // for(auto & col : combinedRowData)
                // {
                //    printf("col data: %s\n", col.c_str());
                // }
                // for(auto & col : columnNameToIndex)
                // {
                //     printf("col name: %s,col index: %d\n", col.first.c_str(), col.second);
                // }
                
                // try
                // {
                // printf("Enteiring condition evaluation\n");
                matches = m_condition.evaluate(colTypes, combinedRowData, columnNameToIndex);
                // }
                // catch (const std::exception &e)
                // {
                //     std::cerr << "Error evaluating join condition: " << e.what() << std::endl;
                //     continue;
                // }

                // If condition matches, add the combined row to the result table
                if (matches)
                {
                    // printf("Row %zu from left table matches row %zu from right table\n", leftRow, rightRow);
                    matchCount++;

                    // Add left table data to result
                    for (size_t col = 0; col < leftColCount; ++col)
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
                        case DataType::STRING:
                        case DataType::VARCHAR:
                            resultTable.appendStringValue(col, m_leftTable.getStringValue(col, leftRow));
                            break;
                        case DataType::DOUBLE:
                            resultTable.appendDoubleValue(col, m_leftTable.getDoubleValue(col, leftRow));
                            break;
                        case DataType::BOOL:
                            resultTable.appendBoolValue(col, m_leftTable.getBoolValue(col, leftRow));
                            break;
                        case DataType::DATE:
                        case DataType::DATETIME:
                            resultTable.appendStringValue(col, m_leftTable.getDateTimeValue(col, leftRow));
                            break;
                        default:
                            throw std::runtime_error("Unsupported data type for join: " + column.getName());
                        }
                    }

                    // Add right table data to result
                    for (size_t col = 0; col < rightColCount; ++col)
                    {
                        size_t resultCol = col + leftColCount;
                        const auto &column = m_rightTable.getColumns()[col];
                        switch (column.getType())
                        {
                        case DataType::INT:
                            resultTable.appendIntValue(resultCol, m_rightTable.getIntValue(col, rightRow));
                            break;
                        case DataType::FLOAT:
                            resultTable.appendFloatValue(resultCol, m_rightTable.getFloatValue(col, rightRow));
                            break;
                        case DataType::STRING:
                        case DataType::VARCHAR:
                            resultTable.appendStringValue(resultCol, m_rightTable.getStringValue(col, rightRow));
                            break;
                        case DataType::DOUBLE:
                            resultTable.appendDoubleValue(resultCol, m_rightTable.getDoubleValue(col, rightRow));
                            break;
                        case DataType::BOOL:
                            resultTable.appendBoolValue(resultCol, m_rightTable.getBoolValue(col, rightRow));
                            break;
                        case DataType::DATE:
                        case DataType::DATETIME:
                            resultTable.appendStringValue(resultCol, m_rightTable.getDateTimeValue(col, rightRow));
                            break;
                        default:
                            throw std::runtime_error("Unsupported data type for join: " + column.getName());
                        }
                    }

                    // Finalize the row
                    resultTable.finalizeRow();
                }
            }
        }

        std::cout << "Joined " << matchCount << " rows from " << leftRowCount << " left rows and "
                  << rightRowCount << " right rows" << std::endl;

        return resultTable;
    }

    Table Join::executeGPU()
    {
        // This would call the GPU kernel implementation
        // Assuming JoinGPU.cuh contains a launchJoinKernel function
        // return launchJoinKernel(m_leftTable, m_rightTable, m_condition);
    }

} // namespace GPUDBMS