#include "../../include/Operations/Select.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <typeinfo>
#include <string>

namespace GPUDBMS
{
    // Select Implementation
    Select::Select(const Table &inputTable, const Condition &condition)
        : m_inputTable(inputTable), m_condition(condition)
    {
    }

    Table Select::execute(bool useGPU)
    {
        if (useGPU)
            return executeGPU();
        else
            return executeCPU();
    }

    Table Select::executeCPU()
    {

        Table resultTable = m_inputTable.createEmptyWithSameSchema();
        const size_t rowCount = m_inputTable.getRowCount();
        const size_t colCount = m_inputTable.getColumnCount();

        std::unordered_map<std::string, int> columnNameToIndex;
        for (size_t i = 0; i < colCount; ++i)
        {
            columnNameToIndex[m_inputTable.getColumns()[i].getName()] = static_cast<int>(i);
        }

        std::vector<DataType> colsType = m_inputTable.getColumnsType();

        // Check if this is likely an unconditional SELECT
        // Get the type name of the condition and look for the word "True"
        const std::string conditionTypeName = typeid(m_condition).name();

        // If condition type name contains "True", it's likely a simple SELECT without WHERE
        if (conditionTypeName.find("True") != std::string::npos) {
            std::cout << "Using optimized path for unconditional select" << std::endl;
            return m_inputTable; // Return the entire table for simple SELECTs
        }

        std::vector<int> includedRows;

        // For each row in the input table
        for (size_t row = 0; row < rowCount; ++row)
        {
            // Extract current row data
            std::vector<std::string> rowData(colCount);
            bool rowHasError = false;
            
            for (size_t col = 0; col < colCount; ++col)
            {
                try {
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
                    case DataType::DATE:
                    case DataType::DATETIME:
                        // Fix: Make sure to get string value for date/datetime
                        rowData[col] = m_inputTable.getDateTimeValue(col, row);
                        break;  
                    default:
                        rowData[col] = ""; // Default for unsupported types
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error getting value for row " << row << ", column " << col 
                              << ": " << e.what() << std::endl;
                    rowData[col] = "";
                    rowHasError = true;
                }
            }

            if (rowHasError) {
                continue; // Skip rows with errors
            }

            // Evaluate condition on this row
            bool matches = false;
            try {
                matches = m_condition.evaluate(colsType, rowData, columnNameToIndex);
            } catch (const std::exception& e) {
                std::cerr << "Error evaluating condition on row " << row << ": " << e.what() << std::endl;
                continue;
            }
            
            if (matches) {
                includedRows.push_back(row);
            }
        }

        std::cout << "Selected " << includedRows.size() << " rows out of " << rowCount << std::endl;
        
        // If no rows matched but we have data, there might be an issue with condition evaluation
        if (includedRows.empty() && rowCount > 0) {
            std::cout << "Warning: No rows matched the selection criteria" << std::endl;
        }

        return m_inputTable.getSlicedTable(includedRows);
    }

    Table Select::executeGPU()
    {
        return launchSelectKernel(m_inputTable, m_condition);
    }

} // namespace GPUDBMS