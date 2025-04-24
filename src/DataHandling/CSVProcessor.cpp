#include "DataHandling/CSVProcessor.hpp"
#include "Utilities/Logger.hpp"
#include "Utilities/ErrorHandling.hpp"
#include "Utilities/StringUtils.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>

namespace SQLQueryProcessor {

std::shared_ptr<Table> CSVProcessor::readCSV(const std::string& filePath) {
    Logger::debug("Reading CSV file: " + filePath);
    
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw DataException("Failed to open CSV file: " + filePath);
    }
    
    // Extract table name from file path
    std::filesystem::path path(filePath);
    std::string tableName = path.stem().string();
    
    auto table = std::make_shared<Table>(tableName);
    
    // Read header line
    std::string headerLine;
    if (!std::getline(file, headerLine)) {
        throw DataException("CSV file is empty: " + filePath);
    }
    
    // Parse header and create columns
    std::vector<std::string> headers = parseHeader(headerLine);
    for (const auto& header : headers) {
        // Parse column metadata
        ColumnMeta meta = parseColumnHeader(header);
        
        // Add column to table
        table->addColumn(meta.name, meta.isPrimary, meta.isForeign);
        
        // Set foreign key reference if applicable
        if (meta.isForeign && !meta.referencedTable.empty()) {
            table->setForeignKeyReference(meta.name, meta.referencedTable);
        }
    }
    
    // Read data rows
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::vector<std::string> rowData = parseLine(line);
        
        // Check if row has the correct number of columns
        if (rowData.size() != headers.size()) {
            Logger::warning("Skipping row with incorrect number of columns in " + filePath);
            continue;
        }
        
        table->addRow(rowData);
    }
    
    Logger::info("Successfully loaded table " + tableName + " with " + 
                std::to_string(table->getRowCount()) + " rows and " + 
                std::to_string(table->getColumnCount()) + " columns");
    
    return table;
}

bool CSVProcessor::writeCSV(const std::shared_ptr<Table>& table, const std::string& filePath) {
    Logger::debug("Writing table to CSV file: " + filePath);
    
    if (!table) {
        Logger::error("Cannot write null table to CSV");
        return false;
    }
    
    std::ofstream file(filePath);
    if (!file.is_open()) {
        Logger::error("Failed to open file for writing: " + filePath);
        return false;
    }
    
    // Write header
    const std::vector<std::string>& columnNames = table->getColumnNames();
    for (size_t i = 0; i < columnNames.size(); ++i) {
        if (i > 0) file << ",";
        
        std::string columnName = columnNames[i];
        
        // Add type indicators
        if (table->isColumnPrimaryKey(i)) {
            columnName += " (N) (P)";  // Assuming primary keys are numeric
        } else {
            // In a real implementation, we would determine the type
            columnName += " (T)";  // Default to text
        }
        
        file << columnName;
    }
    file << std::endl;
    
    // Write data rows
    const std::vector<std::vector<std::string>>& data = table->getData();
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) file << ",";
            
            // Handle values that contain commas or quotes
            std::string value = row[i];
            if (value.find(',') != std::string::npos || 
                value.find('"') != std::string::npos || 
                value.find('\n') != std::string::npos) {
                
                // Escape quotes and wrap in quotes
                std::string escapedValue = StringUtils::replaceAll(value, "\"", "\"\"");
                file << "\"" << escapedValue << "\"";
            } else {
                file << value;
            }
        }
        file << std::endl;
    }
    
    Logger::info("Successfully wrote table " + table->getName() + " to " + filePath);
    return true;
}

std::vector<std::string> CSVProcessor::parseHeader(const std::string& headerLine) {
    return parseLine(headerLine);
}

std::vector<std::string> CSVProcessor::parseLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool inQuotes = false;
    
    for (size_t i = 0; i < line.length(); ++i) {
        char c = line[i];
        
        if (c == '"') {
            if (i + 1 < line.length() && line[i + 1] == '"') {
                // Escaped quote
                field += '"';
                ++i;  // Skip the next quote
            } else {
                // Toggle quote mode
                inQuotes = !inQuotes;
            }
        } else if (c == ',' && !inQuotes) {
            // End of field
            fields.push_back(StringUtils::trim(field));
            field.clear();
        } else {
            field += c;
        }
    }
    
    // Don't forget the last field
    fields.push_back(StringUtils::trim(field));
    
    return fields;
}

CSVProcessor::ColumnMeta CSVProcessor::parseColumnHeader(const std::string& header) {
    ColumnMeta meta;
    std::string remainingHeader = header;
    
    // Extract column type
    size_t typePos = remainingHeader.find(" (");
    if (typePos != std::string::npos) {
        meta.name = StringUtils::trim(remainingHeader.substr(0, typePos));
        remainingHeader = remainingHeader.substr(typePos);
        
        // Determine column type
        if (remainingHeader.find("(N)") != std::string::npos) {
            meta.type = ColumnType::NUMERIC;
        } else if (remainingHeader.find("(D)") != std::string::npos) {
            meta.type = ColumnType::DATETIME;
        } else if (remainingHeader.find("(T)") != std::string::npos) {
            meta.type = ColumnType::TEXT;
        } else {
            meta.type = ColumnType::UNKNOWN;
        }
        
        // Check if it's a primary key
        meta.isPrimary = (remainingHeader.find("(P)") != std::string::npos);
        
        // Check if it's a foreign key reference
        meta.isForeign = isForeignKeyReference(meta.name, meta.referencedTable);
    } else {
        // No type info, use the whole header as the name
        meta.name = StringUtils::trim(header);
        meta.type = ColumnType::TEXT;  // Default to text
        meta.isPrimary = false;
        meta.isForeign = isForeignKeyReference(meta.name, meta.referencedTable);
    }
    
    return meta;
}

bool CSVProcessor::isForeignKeyReference(const std::string& columnName, std::string& referencedTable) {
    // Check if the column name follows the foreign key pattern: tableName_columnName
    size_t underscorePos = columnName.find('_');
    if (underscorePos != std::string::npos) {
        referencedTable = columnName.substr(0, underscorePos);
        return true;
    }
    return false;
}

} // namespace SQLQueryProcessor