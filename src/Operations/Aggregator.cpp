#include "../../include/Operations/Aggregator.hpp"
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <variant>

namespace GPUDBMS
{
    Aggregator::Aggregator(const Table& inputTable, 
                         AggregateFunction function, 
                         const std::string& columnName,
                         const std::optional<std::string>& groupByColumn,
                         const std::string& alias)
        : m_inputTable(inputTable), m_groupByColumn(groupByColumn)
    {
        m_aggregations.push_back(Aggregation(function, columnName, alias));
        resolveGroupByColumnIndex();
    }
    
    Aggregator::Aggregator(const Table& inputTable, 
                         const std::vector<Aggregation>& aggregations,
                         const std::optional<std::string>& groupByColumn)
        : m_inputTable(inputTable), m_aggregations(aggregations), m_groupByColumn(groupByColumn)
    {
        resolveGroupByColumnIndex();
    }
    
    void Aggregator::resolveGroupByColumnIndex()
    {
        if (m_groupByColumn.has_value()) {
            const int index = m_inputTable.getColumnIndex(m_groupByColumn.value());
            if (index < 0) {
                throw std::runtime_error("Group by column not found: " + m_groupByColumn.value());
            }
            m_groupByColumnIndex = index;
        }
    }
    
    Table Aggregator::execute()
    {
        // For now, just call CPU implementation
        // TODO: Implement GPU version
        return executeCPU();
    }
    
    Table Aggregator::executeCPU()
    {
        // Create result table schema
        std::vector<Column> resultColumns;
        
        // Add group by column to the schema if present
        if (m_groupByColumn.has_value()) {
            const int index = m_groupByColumnIndex.value();
            resultColumns.push_back(m_inputTable.getColumns()[index]);
        }
        
        // Add aggregation result columns to the schema
        for (const auto& agg : m_aggregations) {
            // Determine result column type based on aggregation function and input column type
            DataType resultType;
            
            // For COUNT, result is always INT
            if (agg.function == AggregateFunction::COUNT) {
                resultType = DataType::INT;
            } else {
                // Get column index and datatype
                const int colIndex = m_inputTable.getColumnIndex(agg.columnName);
                if (colIndex < 0) {
                    throw std::runtime_error("Column not found for aggregation: " + agg.columnName);
                }
                
                const auto& column = m_inputTable.getColumns()[colIndex];
                DataType inputType = column.getType();
                
                // For SUM and AVG, result is DOUBLE for numeric types
                if (agg.function == AggregateFunction::SUM || agg.function == AggregateFunction::AVG) {
                    if (inputType == DataType::INT || inputType == DataType::FLOAT || inputType == DataType::DOUBLE) {
                        resultType = DataType::DOUBLE;
                    } else {
                        throw std::runtime_error("Cannot apply SUM or AVG to non-numeric column: " + agg.columnName);
                    }
                }
                // For MIN and MAX, result has the same type as the input
                else {
                    resultType = inputType;
                }
            }
            
            resultColumns.push_back(Column(agg.resultName, resultType));
        }
        
        // Create result table
        Table resultTable(resultColumns);
        
        // If grouping, process each group separately
        if (m_groupByColumn.has_value()) {
            const int groupIndex = m_groupByColumnIndex.value();
            const auto& groupCol = m_inputTable.getColumns()[groupIndex];
            
            // Map to store group values and corresponding row indices
            std::unordered_map<std::string, std::vector<size_t>> groups;
            
            // Group rows by the group column values
            for (size_t row = 0; row < m_inputTable.getRowCount(); ++row) {
                std::string groupValue;
                
                // Extract group value based on column type
                switch (groupCol.getType()) {
                    case DataType::INT:
                        groupValue = std::to_string(m_inputTable.getIntValue(groupIndex, row));
                        break;
                    case DataType::FLOAT:
                        groupValue = std::to_string(m_inputTable.getFloatValue(groupIndex, row));
                        break;
                    case DataType::DOUBLE:
                        groupValue = std::to_string(m_inputTable.getDoubleValue(groupIndex, row));
                        break;
                    case DataType::VARCHAR:
                    case DataType::STRING:
                        groupValue = m_inputTable.getStringValue(groupIndex, row);
                        break;
                    case DataType::BOOL:
                        groupValue = m_inputTable.getBoolValue(groupIndex, row) ? "true" : "false";
                        break;
                }
                
                groups[groupValue].push_back(row);
            }
            
            // Process each group
            for (const auto& [groupValue, rowIndices] : groups) {
                // Add group column value
                size_t colIndex = 0;
                
                switch (groupCol.getType()) {
                    case DataType::INT:
                        resultTable.appendIntValue(colIndex, std::stoi(groupValue));
                        break;
                    case DataType::FLOAT:
                        resultTable.appendFloatValue(colIndex, std::stof(groupValue));
                        break;
                    case DataType::DOUBLE:
                        resultTable.appendDoubleValue(colIndex, std::stod(groupValue));
                        break;
                    case DataType::VARCHAR:
                    case DataType::STRING:
                        resultTable.appendStringValue(colIndex, groupValue);
                        break;
                    case DataType::BOOL:
                        resultTable.appendBoolValue(colIndex, groupValue == "true");
                        break;
                }
                colIndex++;
                
                // Apply each aggregation
                for (const auto& agg : m_aggregations) {
                    const int aggColIndex = m_inputTable.getColumnIndex(agg.columnName);
                    if (aggColIndex < 0) {
                        throw std::runtime_error("Aggregation column not found: " + agg.columnName);
                    }
                    
                    const auto& column = m_inputTable.getColumns()[aggColIndex];
                    
                    switch (column.getType()) {
                        case DataType::INT:
                            if (agg.function == AggregateFunction::COUNT) {
                                resultTable.appendIntValue(colIndex, rowIndices.size());
                            } else {
                                // Gather all values for this group
                                std::vector<int> values;
                                for (size_t rowIdx : rowIndices) {
                                    values.push_back(m_inputTable.getIntValue(aggColIndex, rowIdx));
                                }
                                
                                // Apply aggregation
                                switch (agg.function) {
                                    case AggregateFunction::SUM:
                                        resultTable.appendDoubleValue(colIndex, 
                                            std::accumulate(values.begin(), values.end(), 0.0));
                                        break;
                                    case AggregateFunction::AVG:
                                        resultTable.appendDoubleValue(colIndex, 
                                            values.empty() ? 0.0 : 
                                            std::accumulate(values.begin(), values.end(), 0.0) / values.size());
                                        break;
                                    case AggregateFunction::MIN:
                                        resultTable.appendIntValue(colIndex, 
                                            values.empty() ? 0 : *std::min_element(values.begin(), values.end()));
                                        break;
                                    case AggregateFunction::MAX:
                                        resultTable.appendIntValue(colIndex, 
                                            values.empty() ? 0 : *std::max_element(values.begin(), values.end()));
                                        break;
                                    default:
                                        break;
                                }
                            }
                            break;
                            
                        case DataType::FLOAT:
                        case DataType::DOUBLE:
                            if (agg.function == AggregateFunction::COUNT) {
                                resultTable.appendIntValue(colIndex, rowIndices.size());
                            } else {
                                // Gather all values for this group
                                std::vector<double> values;
                                for (size_t rowIdx : rowIndices) {
                                    if (column.getType() == DataType::FLOAT) {
                                        values.push_back(m_inputTable.getFloatValue(aggColIndex, rowIdx));
                                    } else {
                                        values.push_back(m_inputTable.getDoubleValue(aggColIndex, rowIdx));
                                    }
                                }
                                
                                // Apply aggregation
                                switch (agg.function) {
                                    case AggregateFunction::SUM:
                                        resultTable.appendDoubleValue(colIndex, 
                                            std::accumulate(values.begin(), values.end(), 0.0));
                                        break;
                                    case AggregateFunction::AVG:
                                        resultTable.appendDoubleValue(colIndex, 
                                            values.empty() ? 0.0 : 
                                            std::accumulate(values.begin(), values.end(), 0.0) / values.size());
                                        break;
                                    case AggregateFunction::MIN:
                                        resultTable.appendDoubleValue(colIndex, 
                                            values.empty() ? 0.0 : *std::min_element(values.begin(), values.end()));
                                        break;
                                    case AggregateFunction::MAX:
                                        resultTable.appendDoubleValue(colIndex, 
                                            values.empty() ? 0.0 : *std::max_element(values.begin(), values.end()));
                                        break;
                                    default:
                                        break;
                                }
                            }
                            break;
                            
                        case DataType::VARCHAR:
                        case DataType::STRING:
                            if (agg.function == AggregateFunction::COUNT) {
                                resultTable.appendIntValue(colIndex, rowIndices.size());
                            } else if (agg.function == AggregateFunction::MIN || agg.function == AggregateFunction::MAX) {
                                std::vector<std::string> values;
                                for (size_t rowIdx : rowIndices) {
                                    values.push_back(m_inputTable.getStringValue(aggColIndex, rowIdx));
                                }
                                
                                if (!values.empty()) {
                                    if (agg.function == AggregateFunction::MIN) {
                                        resultTable.appendStringValue(colIndex, *std::min_element(values.begin(), values.end()));
                                    } else { // MAX
                                        resultTable.appendStringValue(colIndex, *std::max_element(values.begin(), values.end()));
                                    }
                                } else {
                                    resultTable.appendStringValue(colIndex, "");
                                }
                            } else {
                                throw std::runtime_error("Cannot apply SUM or AVG to non-numeric column: " + agg.columnName);
                            }
                            break;
                            
                        case DataType::BOOL:
                            if (agg.function == AggregateFunction::COUNT) {
                                resultTable.appendIntValue(colIndex, rowIndices.size());
                            } else {
                                throw std::runtime_error("Only COUNT is supported for BOOL columns: " + agg.columnName);
                            }
                            break;
                    }
                    
                    colIndex++;
                }
                
                resultTable.finalizeRow();
            }
        }
        // If not grouping, apply aggregations to the entire table
        else {
            size_t colIndex = 0;
            
            for (const auto& agg : m_aggregations) {
                const int aggColIndex = m_inputTable.getColumnIndex(agg.columnName);
                if (aggColIndex < 0) {
                    throw std::runtime_error("Aggregation column not found: " + agg.columnName);
                }
                
                const auto& column = m_inputTable.getColumns()[aggColIndex];
                
                switch (column.getType()) {
                    case DataType::INT:
                        if (agg.function == AggregateFunction::COUNT) {
                            resultTable.appendIntValue(colIndex, m_inputTable.getRowCount());
                        } else {
                            // Gather all values
                            std::vector<int> values;
                            for (size_t row = 0; row < m_inputTable.getRowCount(); ++row) {
                                values.push_back(m_inputTable.getIntValue(aggColIndex, row));
                            }
                            
                            // Apply aggregation
                            switch (agg.function) {
                                case AggregateFunction::SUM:
                                    resultTable.appendDoubleValue(colIndex, 
                                        std::accumulate(values.begin(), values.end(), 0.0));
                                    break;
                                case AggregateFunction::AVG:
                                    resultTable.appendDoubleValue(colIndex, 
                                        values.empty() ? 0.0 : 
                                        std::accumulate(values.begin(), values.end(), 0.0) / values.size());
                                    break;
                                case AggregateFunction::MIN:
                                    resultTable.appendIntValue(colIndex, 
                                        values.empty() ? 0 : *std::min_element(values.begin(), values.end()));
                                    break;
                                case AggregateFunction::MAX:
                                    resultTable.appendIntValue(colIndex, 
                                        values.empty() ? 0 : *std::max_element(values.begin(), values.end()));
                                    break;
                                default:
                                    break;
                            }
                        }
                        break;
                        
                    case DataType::FLOAT:
                    case DataType::DOUBLE:
                        if (agg.function == AggregateFunction::COUNT) {
                            resultTable.appendIntValue(colIndex, m_inputTable.getRowCount());
                        } else {
                            // Gather all values
                            std::vector<double> values;
                            for (size_t row = 0; row < m_inputTable.getRowCount(); ++row) {
                                if (column.getType() == DataType::FLOAT) {
                                    values.push_back(m_inputTable.getFloatValue(aggColIndex, row));
                                } else {
                                    values.push_back(m_inputTable.getDoubleValue(aggColIndex, row));
                                }
                            }
                            
                            // Apply aggregation
                            switch (agg.function) {
                                case AggregateFunction::SUM:
                                    resultTable.appendDoubleValue(colIndex, 
                                        std::accumulate(values.begin(), values.end(), 0.0));
                                    break;
                                case AggregateFunction::AVG:
                                    resultTable.appendDoubleValue(colIndex, 
                                        values.empty() ? 0.0 : 
                                        std::accumulate(values.begin(), values.end(), 0.0) / values.size());
                                    break;
                                case AggregateFunction::MIN:
                                    resultTable.appendDoubleValue(colIndex, 
                                        values.empty() ? 0.0 : *std::min_element(values.begin(), values.end()));
                                    break;
                                case AggregateFunction::MAX:
                                    resultTable.appendDoubleValue(colIndex, 
                                        values.empty() ? 0.0 : *std::max_element(values.begin(), values.end()));
                                    break;
                                default:
                                    break;
                            }
                        }
                        break;
                        
                    case DataType::VARCHAR:
                    case DataType::STRING:
                        if (agg.function == AggregateFunction::COUNT) {
                            resultTable.appendIntValue(colIndex, m_inputTable.getRowCount());
                        } else if (agg.function == AggregateFunction::MIN || agg.function == AggregateFunction::MAX) {
                            std::vector<std::string> values;
                            for (size_t row = 0; row < m_inputTable.getRowCount(); ++row) {
                                values.push_back(m_inputTable.getStringValue(aggColIndex, row));
                            }
                            
                            if (!values.empty()) {
                                if (agg.function == AggregateFunction::MIN) {
                                    resultTable.appendStringValue(colIndex, *std::min_element(values.begin(), values.end()));
                                } else { // MAX
                                    resultTable.appendStringValue(colIndex, *std::max_element(values.begin(), values.end()));
                                }
                            } else {
                                resultTable.appendStringValue(colIndex, "");
                            }
                        } else {
                            throw std::runtime_error("Cannot apply SUM or AVG to non-numeric column: " + agg.columnName);
                        }
                        break;
                        
                    case DataType::BOOL:
                        if (agg.function == AggregateFunction::COUNT) {
                            resultTable.appendIntValue(colIndex, m_inputTable.getRowCount());
                        } else {
                            throw std::runtime_error("Only COUNT is supported for BOOL columns: " + agg.columnName);
                        }
                        break;
                }
                
                colIndex++;
            }
            
            resultTable.finalizeRow();
        }
        
        return resultTable;
    }
    
} // namespace GPUDBMS