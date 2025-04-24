#include "Operations/Aggregator.hpp"
#include "Utilities/Logger.hpp"
#include "Utilities/ErrorHandling.hpp"
#include "Utilities/StringUtils.hpp"
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <cmath>

namespace SQLQueryProcessor {

void Aggregator::addAggregation(AggregateFunction func, const std::string& columnName, const std::string& alias) {
    aggregations.emplace_back(func, columnName, alias);
}

void Aggregator::addGroupByColumn(const std::string& columnName) {
    groupByColumns.push_back(columnName);
}

void Aggregator::clearAggregations() {
    aggregations.clear();
}

void Aggregator::clearGroupByColumns() {
    groupByColumns.clear();
}

std::shared_ptr<Table> Aggregator::executeCPU(const std::shared_ptr<Table>& inputTable) {
    Logger::debug("Executing CPU aggregation operation");
    
    if (!inputTable) {
        throw ExecutionException("Aggregation operation received null input table");
    }
    
    if (aggregations.empty()) {
        throw ExecutionException("No aggregations specified");
    }
    
    // Create result table
    auto resultTable = std::make_shared<Table>("aggregation_result");
    
    // Add group by columns to result table
    std::vector<int> groupByIndices;
    for (const auto& column : groupByColumns) {
        int colIndex = inputTable->getColumnIndex(column);
        if (colIndex < 0) {
            throw ExecutionException("Group by column not found: " + column);
        }
        groupByIndices.push_back(colIndex);
        resultTable->addColumn(column, false, false);
    }
    
    // Add aggregation columns to result table
    std::vector<int> aggregateIndices;
    for (const auto& aggr : aggregations) {
        int colIndex = -1;
        if (aggr.columnName != "*") {
            colIndex = inputTable->getColumnIndex(aggr.columnName);
            if (colIndex < 0) {
                throw ExecutionException("Aggregation column not found: " + aggr.columnName);
            }
        }
        aggregateIndices.push_back(colIndex);
        resultTable->addColumn(aggr.alias, false, false);
    }
    
    // If no GROUP BY, apply aggregations to the entire table
    if (groupByColumns.empty()) {
        std::vector<std::string> resultRow;
        
        for (size_t i = 0; i < aggregations.size(); ++i) {
            const auto& aggr = aggregations[i];
            int colIndex = aggregateIndices[i];
            
            // Collect all values for the column
            std::vector<std::string> values;
            if (colIndex >= 0) {
                for (size_t j = 0; j < inputTable->getRowCount(); ++j) {
                    values.push_back(inputTable->getValue(j, colIndex));
                }
            } else {
                // For COUNT(*), we don't need column values
                values.resize(inputTable->getRowCount());
            }
            
            // Apply aggregate function
            std::string result = applyAggregateFunction(aggr.function, values);
            resultRow.push_back(result);
        }
        
        resultTable->addRow(resultRow);
    } else {
        // For GROUP BY, use a hash table to group rows
        using GroupKey = std::vector<std::string>;
        struct GroupKeyHash {
            size_t operator()(const GroupKey& key) const {
                size_t hash = 0;
                for (const auto& s : key) {
                    hash ^= std::hash<std::string>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                }
                return hash;
            }
        };
        
        std::unordered_map<GroupKey, std::vector<std::vector<std::string>>, GroupKeyHash> groups;
        
        // Group rows by group by columns
        for (size_t i = 0; i < inputTable->getRowCount(); ++i) {
            GroupKey key;
            for (int colIndex : groupByIndices) {
                key.push_back(inputTable->getValue(i, colIndex));
            }
            
            // Store row for each group
            std::vector<std::string> row;
            for (size_t j = 0; j < inputTable->getColumnCount(); ++j) {
                row.push_back(inputTable->getValue(i, j));
            }
            groups[key].push_back(row);
        }
        
        // Process each group
        for (const auto& [key, rows] : groups) {
            std::vector<std::string> resultRow;
            
            // Add group by values
            resultRow.insert(resultRow.end(), key.begin(), key.end());
            
            // Apply aggregations
            for (size_t i = 0; i < aggregations.size(); ++i) {
                const auto& aggr = aggregations[i];
                int colIndex = aggregateIndices[i];
                
                // Collect values for the column from this group
                std::vector<std::string> values;
                if (colIndex >= 0) {
                    for (const auto& row : rows) {
                        values.push_back(row[colIndex]);
                    }
                } else {
                    // For COUNT(*), we don't need column values
                    values.resize(rows.size());
                }
                
                // Apply aggregate function
                std::string result = applyAggregateFunction(aggr.function, values);
                resultRow.push_back(result);
            }
            
            resultTable->addRow(resultRow);
        }
    }
    
    Logger::debug("CPU aggregation completed, result has " + 
                 std::to_string(resultTable->getRowCount()) + " rows");
    
    return resultTable;
}

std::shared_ptr<Table> Aggregator::executeGPU(const std::shared_ptr<Table>& inputTable) {
    Logger::debug("Executing GPU aggregation operation");
    
    if (!inputTable) {
        throw ExecutionException("Aggregation operation received null input table");
    }
    
    if (aggregations.empty()) {
        throw ExecutionException("No aggregations specified");
    }
    
    // If table is small, use CPU aggregation
    if (inputTable->getRowCount() < 1000) {
        Logger::debug("Small table detected, using CPU aggregation instead");
        return executeCPU(inputTable);
    }
    
    // For now, return CPU implementation
    // In a real implementation, we would use CUDA for aggregation
    Logger::warning("GPU aggregation not fully implemented, falling back to CPU");
    
    return executeCPU(inputTable);
}

std::string Aggregator::applyAggregateFunction(AggregateFunction func, const std::vector<std::string>& values) {
    if (values.empty()) {
        switch (func) {
            case AggregateFunction::COUNT:
                return "0";
            case AggregateFunction::SUM:
            case AggregateFunction::AVG:
                return "0";
            case AggregateFunction::MIN:
            case AggregateFunction::MAX:
                return "";
            default:
                return "";
        }
    }
    
    switch (func) {
        case AggregateFunction::COUNT: {
            // Count non-empty values
            size_t count = 0;
            for (const auto& val : values) {
                if (!val.empty()) {
                    count++;
                }
            }
            return std::to_string(count);
        }
        
        case AggregateFunction::SUM: {
            // Sum numeric values
            double sum = 0.0;
            for (const auto& val : values) {
                if (!val.empty()) {
                    sum += parseNumeric(val);
                }
            }
            return StringUtils::formatDouble(sum, 2);
        }
        
        case AggregateFunction::AVG: {
            // Average numeric values
            double sum = 0.0;
            size_t count = 0;
            for (const auto& val : values) {
                if (!val.empty()) {
                    sum += parseNumeric(val);
                    count++;
                }
            }
            if (count == 0) {
                return "0";
            }
            return StringUtils::formatDouble(sum / count, 2);
        }
        
        case AggregateFunction::MIN: {
            // Minimum value
            double minVal = std::numeric_limits<double>::max();
            bool foundValue = false;
            
            for (const auto& val : values) {
                if (!val.empty()) {
                    double num = parseNumeric(val);
                    if (!foundValue || num < minVal) {
                        minVal = num;
                        foundValue = true;
                    }
                }
            }
            
            if (!foundValue) {
                return "";
            }
            
            // Check if the min value is an integer
            if (std::abs(minVal - std::round(minVal)) < 1e-6) {
                return std::to_string(static_cast<int>(minVal));
            } else {
                return StringUtils::formatDouble(minVal, 2);
            }
        }
        
        case AggregateFunction::MAX: {
            // Maximum value
            double maxVal = std::numeric_limits<double>::lowest();
            bool foundValue = false;
            
            for (const auto& val : values) {
                if (!val.empty()) {
                    double num = parseNumeric(val);
                    if (!foundValue || num > maxVal) {
                        maxVal = num;
                        foundValue = true;
                    }
                }
            }
            
            if (!foundValue) {
                return "";
            }
            
            // Check if the max value is an integer
            if (std::abs(maxVal - std::round(maxVal)) < 1e-6) {
                return std::to_string(static_cast<int>(maxVal));
            } else {
                return StringUtils::formatDouble(maxVal, 2);
            }
        }
        
        default:
            return "";
    }
}

double Aggregator::parseNumeric(const std::string& value) {
    try {
        return std::stod(value);
    } catch (...) {
        return 0.0;
    }
}

std::string Aggregator::AggregateInfo::generateDefaultAlias(AggregateFunction func, const std::string& col) {
    std::string funcName;
    switch (func) {
        case AggregateFunction::COUNT: funcName = "COUNT"; break;
        case AggregateFunction::SUM:   funcName = "SUM"; break;
        case AggregateFunction::AVG:   funcName = "AVG"; break;
        case AggregateFunction::MIN:   funcName = "MIN"; break;
        case AggregateFunction::MAX:   funcName = "MAX"; break;
        default:                       funcName = "UNKNOWN"; break;
    }
    
    return funcName + "(" + col + ")";
}

} // namespace SQLQueryProcessor