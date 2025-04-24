#include "Operations/OrderBy.hpp"
#include "Utilities/Logger.hpp"
#include "Utilities/ErrorHandling.hpp"
#include "Utilities/StringUtils.hpp"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

namespace SQLQueryProcessor {

void OrderBy::addSortKey(const std::string& columnName, SortOrder order) {
    // Determine sort type automatically from data or schema info
    SortType type = SortType::TEXT;  // Default
    sortKeys.emplace_back(columnName, order, type);
}

void OrderBy::clearSortKeys() {
    sortKeys.clear();
}

std::shared_ptr<Table> OrderBy::executeCPU(const std::shared_ptr<Table>& inputTable) {
    Logger::debug("Executing CPU order by operation with " + 
                 std::to_string(sortKeys.size()) + " sort key(s)");
    
    if (!inputTable) {
        throw ExecutionException("OrderBy operation received null input table");
    }
    
    if (sortKeys.empty()) {
        // No sort keys, return a copy of the input table
        auto resultTable = inputTable->createSimilarTable("orderby_result");
        for (size_t i = 0; i < inputTable->getRowCount(); ++i) {
            resultTable->addRow(inputTable->getData()[i]);
        }
        return resultTable;
    }
    
    // Create output table with same schema
    auto resultTable = inputTable->createSimilarTable("orderby_result");
    
    // Copy all data
    auto sortableData = inputTable->getData();
    
    // Prepare column indices and sort orders
    std::vector<int> keyIndices;
    std::vector<SortOrder> orders;
    std::vector<SortType> types;
    
    for (const auto& key : sortKeys) {
        int columnIndex = inputTable->getColumnIndex(key.columnName);
        if (columnIndex < 0) {
            throw ExecutionException("Sort column not found: " + key.columnName);
        }
        
        keyIndices.push_back(columnIndex);
        orders.push_back(key.order);
        
        // Determine sort type based on column data
        SortType type = determineSortType(inputTable, key.columnName);
        types.push_back(type);
    }
    
    // Perform quick sort
    quickSortCPU(sortableData, 0, sortableData.size() - 1, keyIndices, orders, types);
    
    // Add sorted data to result table
    for (const auto& row : sortableData) {
        resultTable->addRow(row);
    }
    
    Logger::debug("CPU orderby operation completed");
    
    return resultTable;
}

std::shared_ptr<Table> OrderBy::executeGPU(const std::shared_ptr<Table>& inputTable) {
    Logger::debug("Executing GPU order by operation with " + 
                 std::to_string(sortKeys.size()) + " sort key(s)");
    
    if (!inputTable) {
        throw ExecutionException("OrderBy operation received null input table");
    }
    
    if (sortKeys.empty()) {
        // No sort keys, return a copy of the input table
        auto resultTable = inputTable->createSimilarTable("orderby_result");
        for (size_t i = 0; i < inputTable->getRowCount(); ++i) {
            resultTable->addRow(inputTable->getData()[i]);
        }
        return resultTable;
    }
    
    // For small tables, CPU might be faster
    if (inputTable->getRowCount() < 10000) {
        Logger::debug("Table too small for GPU sorting, falling back to CPU");
        return executeCPU(inputTable);
    }
    
    // Create output table with same schema
    auto resultTable = inputTable->createSimilarTable("orderby_result");
    
    // Prepare column indices and sort orders
    std::vector<int> keyIndices;
    std::vector<SortOrder> orders;
    std::vector<SortType> types;
    
    for (const auto& key : sortKeys) {
        int columnIndex = inputTable->getColumnIndex(key.columnName);
        if (columnIndex < 0) {
            throw ExecutionException("Sort column not found: " + key.columnName);
        }
        
        keyIndices.push_back(columnIndex);
        orders.push_back(key.order);
        
        // Determine sort type based on column data
        SortType type = determineSortType(inputTable, key.columnName);
        types.push_back(type);
    }
    
    // In a real implementation, we would perform sorting on the GPU
    // For now, as a placeholder, fall back to CPU sorting
    
    auto sortableData = inputTable->getData();
    quickSortCPU(sortableData, 0, sortableData.size() - 1, keyIndices, orders, types);
    
    // Add sorted data to result table
    for (const auto& row : sortableData) {
        resultTable->addRow(row);
    }
    
    Logger::debug("GPU orderby operation completed (CPU fallback)");
    
    return resultTable;
}

SortType OrderBy::determineSortType(const std::shared_ptr<Table>& table, const std::string& columnName) {
    int columnIndex = table->getColumnIndex(columnName);
    if (columnIndex < 0) {
        return SortType::TEXT;  // Default
    }
    
    // Sample some rows to determine type
    const size_t sampleSize = std::min(table->getRowCount(), static_cast<size_t>(100));
    int numericCount = 0;
    int datetimeCount = 0;
    
    for (size_t i = 0; i < sampleSize; ++i) {
        const std::string& value = table->getValue(i, columnIndex);
        
        // Try to parse as numeric
        double numericValue;
        if (StringUtils::tryParseDouble(value, numericValue)) {
            numericCount++;
            continue;
        }
        
        // Check if it's a datetime (would have a more sophisticated check in reality)
        if (value.length() >= 10 && value.find('-') != std::string::npos) {
            datetimeCount++;
        }
    }
    
    // Determine most likely type
    if (numericCount > sampleSize / 2) {
        return SortType::NUMERIC;
    } else if (datetimeCount > sampleSize / 2) {
        return SortType::DATETIME;
    } else {
        return SortType::TEXT;
    }
}

void OrderBy::quickSortCPU(std::vector<std::vector<std::string>>& data, int low, int high, 
                          const std::vector<int>& keyIndices, const std::vector<SortOrder>& orders,
                          const std::vector<SortType>& types) {
    if (low < high) {
        int pivotIndex = partitionCPU(data, low, high, keyIndices, orders, types);
        quickSortCPU(data, low, pivotIndex - 1, keyIndices, orders, types);
        quickSortCPU(data, pivotIndex + 1, high, keyIndices, orders, types);
    }
}

int OrderBy::partitionCPU(std::vector<std::vector<std::string>>& data, int low, int high,
                         const std::vector<int>& keyIndices, const std::vector<SortOrder>& orders,
                         const std::vector<SortType>& types) {
    // Choose the rightmost element as pivot
    const auto& pivot = data[high];
    int i = low - 1;
    
    for (int j = low; j < high; j++) {
        // Compare based on sort keys
        int compareResult = compareValues(data[j][keyIndices[0]], pivot[keyIndices[0]], 
                                         types[0], orders[0]);
        
        // If current element is less than or equal to pivot
        if (compareResult <= 0) {
            i++;
            std::swap(data[i], data[j]);
        }
    }
    
    // Swap pivot to its correct position
    std::swap(data[i + 1], data[high]);
    return i + 1;
}

int OrderBy::compareValues(const std::string& val1, const std::string& val2, 
                          SortType type, SortOrder order) {
    int result = 0;
    
    switch (type) {
        case SortType::NUMERIC: {
            double num1, num2;
            bool valid1 = StringUtils::tryParseDouble(val1, num1);
            bool valid2 = StringUtils::tryParseDouble(val2, num2);
            
            if (valid1 && valid2) {
                result = (num1 < num2) ? -1 : ((num1 > num2) ? 1 : 0);
            } else {
                // Fall back to string comparison if parsing fails
                result = val1.compare(val2);
            }
            break;
        }
        
        case SortType::DATETIME: {
            // In a real implementation, this would properly parse and compare dates
            // For now, use string comparison which works for ISO format dates
            result = val1.compare(val2);
            break;
        }
        
        case SortType::TEXT:
        default:
            result = val1.compare(val2);
            break;
    }
    
    // Adjust for sort order
    return (order == SortOrder::ASCENDING) ? result : -result;
}

// CUDA kernel implementations would go in a .cu file
/*
__global__ void bitonicSortKernel(char** data, int* keyIndices, int* sortOrders, 
                                int* sortTypes, int numRows, int numColumns, 
                                int numKeys, int step, int stage) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRows / 2) {
        // Calculate actual position
        int pos = idx + (idx / step) * step;
        int partner = pos + (stage == 0 ? step : -step);
        
        if (partner < numRows) {
            // Compare rows based on sort keys
            bool shouldSwap = false;
            
            for (int k = 0; k < numKeys; k++) {
                int keyIdx = keyIndices[k];
                int sortType = sortTypes[k];
                int sortOrder = sortOrders[k];
                
                const char* val1 = data[pos * numColumns + keyIdx];
                const char* val2 = data[partner * numColumns + keyIdx];
                
                int compareResult = 0;
                
                // Compare based on type
                if (sortType == 0) {  // NUMERIC
                    double num1 = strtod(val1, nullptr);
                    double num2 = strtod(val2, nullptr);
                    compareResult = (num1 < num2) ? -1 : ((num1 > num2) ? 1 : 0);
                } else {  // TEXT or DATETIME
                    compareResult = strcmp(val1, val2);
                }
                
                // Adjust for sort order
                if (sortOrder == 1)  // DESCENDING
                    compareResult = -compareResult;
                
                if (compareResult != 0) {
                    shouldSwap = (compareResult > 0);
                    break;
                }
            }
            
            // Swap if needed
            if (shouldSwap) {
                for (int c = 0; c < numColumns; c++) {
                    char* temp = data[pos * numColumns + c];
                    data[pos * numColumns + c] = data[partner * numColumns + c];
                    data[partner * numColumns + c] = temp;
                }
            }
        }
    }
}
*/

} // namespace SQLQueryProcessor