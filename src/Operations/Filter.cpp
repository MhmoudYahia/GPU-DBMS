// #include "../../include/Operations/Filter.hpp"
// #include <iostream>
// #include <vector>
// #include <algorithm>

// namespace GPUDBMS
// {

//     Filter::Filter(const Table &inputTable, const std::vector<std::unique_ptr<Condition>> &conditions, bool isAnd)
//         : m_inputTable(inputTable), m_isAnd(isAnd)
//     {
//         // Clone each condition to store a deep copy
//         for (const auto &cond : conditions)
//         {
//             m_conditions.push_back(cond->clone());
//         }
//     }

//     Table Filter::execute()
//     {
//         // For now, just call CPU implementation
//         // TODO: Implement GPU version
//         return executeCPU();
//     }

//     Table Filter::executeCPU()
//     {
//         Table resultTable = m_inputTable.createEmptyWithSameSchema();
//         const size_t rowCount = m_inputTable.getRowCount();
//         const size_t colCount = m_inputTable.getColumnCount();

//         std::unordered_map<std::string, int> columnNameToIndex;
//         for (size_t i = 0; i < colCount; ++i)
//         {
//             columnNameToIndex[m_inputTable.getColumns()[i].getName()] = static_cast<int>(i);
//         }

//         std::vector<DataType> colsType = m_inputTable.getColumnsType();

//         // For each row in the input table
//         for (size_t row = 0; row < rowCount; ++row)
//         {
//             // Extract current row data
//             std::vector<std::string> rowData(colCount);
//             for (size_t col = 0; col < colCount; ++col)
//             {
//                 // Get column data based on column type
//                 const auto &column = m_inputTable.getColumns()[col];
//                 switch (column.getType())
//                 {
//                 case DataType::INT:
//                     rowData[col] = std::to_string(m_inputTable.getIntValue(col, row));
//                     break;
//                 case DataType::FLOAT:
//                     rowData[col] = std::to_string(m_inputTable.getFloatValue(col, row));
//                     break;
//                 case DataType::STRING:
//                     rowData[col] = m_inputTable.getStringValue(col, row);
//                     break;
//                 case DataType::VARCHAR:
//                     rowData[col] = m_inputTable.getStringValue(col, row);
//                     break;
//                 case DataType::DOUBLE:
//                     rowData[col] = std::to_string(m_inputTable.getDoubleValue(col, row));
//                     break;
//                 case DataType::BOOL:
//                     rowData[col] = m_inputTable.getBoolValue(col, row) ? "true" : "false";
//                     break;
//                 default:
//                     rowData[col] = ""; // Default for unsupported types
//                 }
//             }

//             // Evaluate all conditions on this row
//             bool rowSatisfiesConditions;

//             if (m_isAnd)
//             {
//                 // For AND logic, all conditions must be satisfied
//                 rowSatisfiesConditions = true;
//                 for (const auto &condition : m_conditions)
//                 {
//                     if (!condition->evaluate(colsType, rowData, columnNameToIndex))
//                     {
//                         rowSatisfiesConditions = false;
//                         break;
//                     }
//                 }
//             }
//             else
//             {
//                 // For OR logic, at least one condition must be satisfied
//                 rowSatisfiesConditions = false;
//                 for (const auto &condition : m_conditions)
//                 {
//                     if (condition->evaluate(colsType, rowData, columnNameToIndex))
//                     {
//                         rowSatisfiesConditions = true;
//                         break;
//                     }
//                 }
//             }

//             // Add the row to result table if it satisfies the combined conditions
//             if (rowSatisfiesConditions)
//             {
//                 for (size_t col = 0; col < colCount; ++col)
//                 {
//                     const auto &column = m_inputTable.getColumns()[col];
//                     switch (column.getType())
//                     {
//                     case DataType::INT:
//                         resultTable.appendIntValue(col, m_inputTable.getIntValue(col, row));
//                         break;
//                     case DataType::FLOAT:
//                         resultTable.appendFloatValue(col, m_inputTable.getFloatValue(col, row));
//                         break;
//                     case DataType::STRING:
//                         resultTable.appendStringValue(col, m_inputTable.getStringValue(col, row));
//                         break;
//                     case DataType::VARCHAR:
//                         resultTable.appendStringValue(col, m_inputTable.getStringValue(col, row));
//                         break;
//                     case DataType::DOUBLE:
//                         resultTable.appendDoubleValue(col, m_inputTable.getDoubleValue(col, row));
//                         break;
//                     case DataType::BOOL:
//                         resultTable.appendBoolValue(col, m_inputTable.getBoolValue(col, row));
//                         break;
//                     default:
//                         // Handle default case or ignore
//                         break;
//                     }
//                 }
//                 resultTable.finalizeRow();
//             }
//         }

//         return resultTable;
//     }

// } // namespace GPUDBMS