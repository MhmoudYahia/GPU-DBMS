#pragma once

#include "Table.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace GPUDBMS {

enum class CSVColumnType {
    NUMBER,  // (N)
    TEXT,    // (T)
    DATE,    // (D)
    BOOLEAN, // (B)
    UNKNOWN
};

class CSVProcessor {
public:
    CSVProcessor() = default;
    
    // Read a CSV file into a Table
    Table readCSV(const std::string& filePath);
    
    // Write a Table to a CSV file
    void writeCSV(const Table& table, const std::string& filePath, bool preserveTypeAnnotations = true);
    
private:
    // Parse a CSV header with type annotations like (N), (T), etc.
    std::vector<Column> parseHeaderWithTypeAnnotations(const std::string& headerLine);
    
    // Determine if a column is a primary key from its header
    bool isPrimaryKey(const std::string& header);
    
    // Extract the base column name from a header with annotations
    std::string extractColumnName(const std::string& header);
    
    // Extract type annotations from a header
    std::vector<std::string> extractTypeAnnotations(const std::string& header);
    
    // Map CSV type annotations to DataType
    DataType mapAnnotationToDataType(const std::vector<std::string>& annotations);
    
    // Split a CSV line into values
    std::vector<std::string> splitCSVLine(const std::string& line);
    
    // Add a row of values to a table
    void addRowToTable(Table& table, const std::vector<std::string>& values);
    
    // Escape special characters for CSV output
    std::string escapeCSVString(const std::string& str);
    
    // Format a column header with type annotations for writing
    std::string formatHeaderWithTypeAnnotations(const std::string& columnName, DataType type, bool isPrimary);
    
    // Map from column name to original header with annotations
    std::unordered_map<std::string, std::string> m_originalHeaders;
};

} // namespace GPUDBMS