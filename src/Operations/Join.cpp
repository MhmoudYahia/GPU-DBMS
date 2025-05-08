#include "../../include/Operations/Join.hpp"
#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace GPUDBMS
{
    Join::Join(const Table &leftTable, const Table &rightTable, const Condition &condition, JoinType joinType)
        : m_leftTable(leftTable), m_rightTable(rightTable), m_condition(condition), m_joinType(joinType)
    {
    }

    Table Join::execute(bool useGPU)
    {
        if (useGPU && false) // Placeholder for GPU implementation
            return executeGPU();
        else
            return executeCPU();
    }

    Table Join::executeCPU()
    {
        std::cout << "Executing " << getJoinTypeName() << " join operation on CPU..." << std::endl;
        
        // Create result table schema by combining both tables
        std::vector<Column> resultColumns;
        
        // Add columns from left table
        for (const auto &column : m_leftTable.getColumns())
        {
            resultColumns.push_back(column);
        }
        
        // Add columns from right table (ensuring no name conflicts)
        for (const auto &column : m_rightTable.getColumns())
        {
            // Check if this column name already exists in result
            std::string columnName = column.getName();
            bool nameExists = false;
            
            for (const auto &existingCol : resultColumns)
            {
                if (existingCol.getName() == columnName)
                {
                    nameExists = true;
                    break;
                }
            }
            
            // If name exists, append prefix to make it unique
            if (nameExists)
            {
                columnName = "right_" + columnName;
            }
            
            resultColumns.push_back(Column(columnName, column.getType()));
        }
        
        // Create the result table
        Table resultTable(resultColumns);
        
        // Prepare column types and indices for condition evaluation
        std::vector<DataType> columnsType;
        std::unordered_map<std::string, int> columnNameToIndex;
        
        // Build combined list of column types and name->index mapping
        for (size_t i = 0; i < resultColumns.size(); i++)
        {
            columnsType.push_back(resultColumns[i].getType());
            columnNameToIndex[resultColumns[i].getName()] = static_cast<int>(i);
        }
        
        // Add mappings for table-qualified column names (for joins)
        for (size_t i = 0; i < m_leftTable.getColumnCount(); i++) {
            // Create mappings for left table columns with table alias
            std::string colName = m_leftTable.getColumnName(i);
            columnNameToIndex["o." + colName] = static_cast<int>(i);
        }
        
        for (size_t i = 0; i < m_rightTable.getColumnCount(); i++) {
            // Create mappings for right table columns with table alias
            std::string colName = m_rightTable.getColumnName(i);
            int rightIndex = static_cast<int>(m_leftTable.getColumnCount() + i);
            
            // Use the original name for the alias mapping
            columnNameToIndex["p." + colName] = rightIndex;
        }
        
        // For each row in left table
        for (size_t leftRow = 0; leftRow < m_leftTable.getRowCount(); leftRow++)
        {
            // For each row in right table
            for (size_t rightRow = 0; rightRow < m_rightTable.getRowCount(); rightRow++)
            {
                // Prepare a combined row for condition evaluation
                std::vector<std::string> combinedRowData(resultColumns.size());
                
                // Fill with left table data
                for (size_t leftCol = 0; leftCol < m_leftTable.getColumnCount(); leftCol++)
                {
                    const auto &column = m_leftTable.getColumns()[leftCol];
                    switch (column.getType())
                    {
                    case DataType::INT:
                        combinedRowData[leftCol] = std::to_string(m_leftTable.getIntValue(leftCol, leftRow));
                        break;
                    case DataType::FLOAT:
                        combinedRowData[leftCol] = std::to_string(m_leftTable.getFloatValue(leftCol, leftRow));
                        break;
                    case DataType::DOUBLE:
                        combinedRowData[leftCol] = std::to_string(m_leftTable.getDoubleValue(leftCol, leftRow));
                        break;
                    case DataType::STRING:
                    case DataType::VARCHAR:
                        combinedRowData[leftCol] = m_leftTable.getStringValue(leftCol, leftRow);
                        break;
                    case DataType::BOOL:
                        combinedRowData[leftCol] = m_leftTable.getBoolValue(leftCol, leftRow) ? "true" : "false";
                        break;
                    case DataType::DATE:
                    case DataType::DATETIME:
                        combinedRowData[leftCol] = m_leftTable.getDateTimeValue(leftCol, leftRow);
                        break;
                    default:
                        combinedRowData[leftCol] = "";
                    }
                }
                
                // Fill with right table data
                for (size_t rightCol = 0; rightCol < m_rightTable.getColumnCount(); rightCol++)
                {
                    size_t resultCol = m_leftTable.getColumnCount() + rightCol;
                    const auto &column = m_rightTable.getColumns()[rightCol];
                    switch (column.getType())
                    {
                    case DataType::INT:
                        combinedRowData[resultCol] = std::to_string(m_rightTable.getIntValue(rightCol, rightRow));
                        break;
                    case DataType::FLOAT:
                        combinedRowData[resultCol] = std::to_string(m_rightTable.getFloatValue(rightCol, rightRow));
                        break;
                    case DataType::DOUBLE:
                        combinedRowData[resultCol] = std::to_string(m_rightTable.getDoubleValue(rightCol, rightRow));
                        break;
                    case DataType::STRING:
                    case DataType::VARCHAR:
                        combinedRowData[resultCol] = m_rightTable.getStringValue(rightCol, rightRow);
                        break;
                    case DataType::BOOL:
                        combinedRowData[resultCol] = m_rightTable.getBoolValue(rightCol, rightRow) ? "true" : "false";
                        break;
                    case DataType::DATE:
                    case DataType::DATETIME:
                        combinedRowData[resultCol] = m_rightTable.getDateTimeValue(rightCol, rightRow);
                        break;
                    default:
                        combinedRowData[resultCol] = "";
                    }
                }
                
                // Evaluate join condition
                bool rowMatches = m_condition.evaluate(columnsType, combinedRowData, columnNameToIndex);
                
                if (rowMatches)
                {
                    // Add matched row to result table
                    for (size_t leftCol = 0; leftCol < m_leftTable.getColumnCount(); leftCol++)
                    {
                        const auto &column = m_leftTable.getColumns()[leftCol];
                        switch (column.getType())
                        {
                        case DataType::INT:
                            resultTable.appendIntValue(leftCol, m_leftTable.getIntValue(leftCol, leftRow));
                            break;
                        case DataType::FLOAT:
                            resultTable.appendFloatValue(leftCol, m_leftTable.getFloatValue(leftCol, leftRow));
                            break;
                        case DataType::DOUBLE:
                            resultTable.appendDoubleValue(leftCol, m_leftTable.getDoubleValue(leftCol, leftRow));
                            break;
                        case DataType::STRING:
                        case DataType::VARCHAR:
                            resultTable.appendStringValue(leftCol, m_leftTable.getStringValue(leftCol, leftRow));
                            break;
                        case DataType::BOOL:
                            resultTable.appendBoolValue(leftCol, m_leftTable.getBoolValue(leftCol, leftRow));
                            break;
                        case DataType::DATE:
                        case DataType::DATETIME:
                            resultTable.appendStringValue(leftCol, m_leftTable.getDateTimeValue(leftCol, leftRow));
                            break;
                        }
                    }
                    
                    // Add right table values
                    for (size_t rightCol = 0; rightCol < m_rightTable.getColumnCount(); rightCol++)
                    {
                        size_t resultCol = m_leftTable.getColumnCount() + rightCol;
                        const auto &column = m_rightTable.getColumns()[rightCol];
                        switch (column.getType())
                        {
                        case DataType::INT:
                            resultTable.appendIntValue(resultCol, m_rightTable.getIntValue(rightCol, rightRow));
                            break;
                        case DataType::FLOAT:
                            resultTable.appendFloatValue(resultCol, m_rightTable.getFloatValue(rightCol, rightRow));
                            break;
                        case DataType::DOUBLE:
                            resultTable.appendDoubleValue(resultCol, m_rightTable.getDoubleValue(rightCol, rightRow));
                            break;
                        case DataType::STRING:
                        case DataType::VARCHAR:
                            resultTable.appendStringValue(resultCol, m_rightTable.getStringValue(rightCol, rightRow));
                            break;
                        case DataType::BOOL:
                            resultTable.appendBoolValue(resultCol, m_rightTable.getBoolValue(rightCol, rightRow));
                            break;
                        case DataType::DATE:
                        case DataType::DATETIME:
                            resultTable.appendStringValue(resultCol, m_rightTable.getDateTimeValue(rightCol, rightRow));
                            break;
                        }
                    }
                    
                    resultTable.finalizeRow();
                }
            }
        }
        
        std::cout << "Join completed: " << resultTable.getRowCount() << " rows in result" << std::endl;
        return resultTable;
    } 
    Table Join::executeGPU()
    {
        // Not implemented yet
        std::cout << "GPU Join not implemented yet, falling back to CPU implementation" << std::endl;
        return executeCPU();
    }

    std::string Join::getJoinTypeName() const
    {
        switch (m_joinType)
        {
        case JoinType::INNER:
            return "INNER";
        case JoinType::LEFT:
            return "LEFT OUTER";
        case JoinType::RIGHT:
            return "RIGHT OUTER";
        case JoinType::FULL:
            return "FULL OUTER";
        default:
            return "UNKNOWN";
        }
    }
} // namespace GPUDBMS