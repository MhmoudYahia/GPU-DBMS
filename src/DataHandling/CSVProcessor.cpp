#include "../../include/DataHandling/StorageManager.hpp"
#include "../../include/DataHandling/CSVProcessor.hpp"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <regex>
#include <stdexcept>

namespace GPUDBMS {

StorageManager::StorageManager(const std::string& dataDirectory)
    : m_dataDirectory(dataDirectory)
{
    // Ensure the data directory exists
    std::filesystem::create_directories(dataDirectory);
    std::filesystem::create_directories(dataDirectory + "/outputs/csv");
    std::filesystem::create_directories(dataDirectory + "/outputs/txt");
}

StorageManager::~StorageManager() {
}

Table CSVProcessor::readCSV(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + filePath);
    }
    
    std::string headerLine;
    
    // Read the header line
    if (!std::getline(file, headerLine)) {
        throw std::runtime_error("CSV file is empty or couldn't read header: " + filePath);
    }
    
    // Parse headers with type annotations
    std::vector<Column> columns = parseHeaderWithTypeAnnotations(headerLine);
    Table table(columns);
    
    // Process data rows
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::vector<std::string> values = splitCSVLine(line);
        
        if (values.size() != columns.size()) {
            // Log warning and skip this row
            std::cerr << "Warning: Skipping row with incorrect number of values: " << line << std::endl;
            continue;
        }
        
        addRowToTable(table, values);
        table.finalizeRow();
    }
    
    return table;
}

void CSVProcessor::writeCSV(const Table& table, const std::string& filePath, bool preserveTypeAnnotations) {
    std::ofstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file for writing: " + filePath);
    }
    
    const std::vector<Column>& columns = table.getColumns();
    
    // Write the header
    for (size_t i = 0; i < columns.size(); ++i) {
        if (i > 0) file << ",";
        
        const std::string& columnName = columns[i].getName();
        
        if (preserveTypeAnnotations) {
            // Check if we have the original header stored
            auto it = m_originalHeaders.find(columnName);
            if (it != m_originalHeaders.end()) {
                file << it->second; // Use original header with annotations
            } else {
                // Create header with type annotations
                bool isPrimary = false; // Would need to check metadata for this
                file << formatHeaderWithTypeAnnotations(columnName, columns[i].getType(), isPrimary);
            }
        } else {
            file << columnName;
        }
    }
    file << std::endl;
    
    // Write each row
    for (size_t row = 0; row < table.getRowCount(); ++row) {
        for (size_t col = 0; col < columns.size(); ++col) {
            if (col > 0) file << ",";
            
            // Format based on column type
            switch (columns[col].getType()) {
                case DataType::INT:
                    file << table.getIntValue(col, row);
                    break;
                case DataType::FLOAT:
                    file << table.getFloatValue(col, row);
                    break;
                case DataType::DOUBLE:
                    file << table.getDoubleValue(col, row);
                    break;
                case DataType::VARCHAR:
                case DataType::STRING:
                    file << "\"" << escapeCSVString(table.getStringValue(col, row)) << "\"";
                    break;
                case DataType::BOOL:
                    file << (table.getBoolValue(col, row) ? "true" : "false");
                    break;
                case DataType::DATE:
                    // Format date values properly
                    file << "\"" << table.getStringValue(col, row) << "\"";
                    break;
                case DataType::DATETIME:
                    // Add proper DateTime formatting - using quotes to preserve the format
                    file << "\"" << escapeCSVString(table.getStringValue(col, row)) << "\"";
                    break;
                default:
                    file << "NULL";
            }
        }
        file << std::endl;
    }
    
    file.close();
}

std::vector<Column> CSVProcessor::parseHeaderWithTypeAnnotations(const std::string& headerLine) {
    std::vector<std::string> headers = splitCSVLine(headerLine);
    std::vector<Column> columns;
    
    // Ensure we have at least some headers
    if (headers.empty()) {
        throw std::runtime_error("No headers found in CSV");
    }
    
    // Process each header with its type annotations
    for (const std::string& header : headers) {
        // Clean up the header (remove quotes)
        std::string cleanHeader = header;
        if (cleanHeader.front() == '"' && cleanHeader.back() == '"') {
            cleanHeader = cleanHeader.substr(1, cleanHeader.length() - 2);
        }
        
        // Extract the column name and type annotations
        std::string columnName = extractColumnName(cleanHeader);
        std::vector<std::string> typeAnnotations = extractTypeAnnotations(cleanHeader);
        
        // Map type annotations to DataType
        DataType dataType = mapAnnotationToDataType(typeAnnotations);
        
        // Create column and add to the list
        columns.push_back(Column(columnName, dataType));
        
        // Store original header with annotations for later use when writing
        m_originalHeaders[columnName] = cleanHeader;
    }
    
    return columns;
}

bool CSVProcessor::isPrimaryKey(const std::string& header) {
    return header.find("(P)") != std::string::npos;
}

std::string CSVProcessor::extractColumnName(const std::string& header) {
    // Extract base column name before type annotations
    std::size_t firstParen = header.find(" (");
    if (firstParen != std::string::npos) {
        return header.substr(0, firstParen);
    }
    
    // If no annotations, return the whole header
    return header;
}

std::vector<std::string> CSVProcessor::extractTypeAnnotations(const std::string& header) {
    std::vector<std::string> annotations;
    std::regex annotationPattern("\\(([^)]+)\\)");
    
    std::string::const_iterator searchStart(header.begin());
    std::smatch match;
    
    while (std::regex_search(searchStart, header.end(), match, annotationPattern)) {
        annotations.push_back(match[1].str());
        searchStart = match.suffix().first;
    }
    
    return annotations;
}

DataType CSVProcessor::mapAnnotationToDataType(const std::vector<std::string>& annotations) {
    // Default to VARCHAR if no annotations
    if (annotations.empty()) {
        return DataType::VARCHAR;
    }
    
    // Check for type annotations
    for (const auto& annotation : annotations) {
        if (annotation == "N") {
            // Could be INT or DOUBLE - choose INT as default
            return DataType::DOUBLE;
        }
        else if (annotation == "T") {
            return DataType::VARCHAR;
        }
        else if (annotation == "B") {
            return DataType::BOOL;
        }
        else if (annotation == "D") {
            return DataType::DATETIME; // New annotation for DateTime
        }
    }
    
    // Default to VARCHAR
    return DataType::VARCHAR;
}

std::vector<std::string> CSVProcessor::splitCSVLine(const std::string& line) {
    std::vector<std::string> values;
    std::string currentValue;
    bool inQuotes = false;
    
    for (size_t i = 0; i < line.length(); ++i) {
        char c = line[i];
        
        if (c == '"') {
            // Handle quotes - they can be escaped by doubling
            if (i + 1 < line.length() && line[i + 1] == '"') {
                currentValue += '"';
                ++i; // Skip the next quote
            } else {
                inQuotes = !inQuotes;
            }
        } else if (c == ',' && !inQuotes) {
            // End of a value
            values.push_back(currentValue);
            currentValue.clear();
        } else {
            currentValue += c;
        }
    }
    
    // Add the last value
    values.push_back(currentValue);
    
    return values;
}

void CSVProcessor::addRowToTable(Table& table, const std::vector<std::string>& values) {
    for (size_t col = 0; col < values.size() && col < table.getColumnCount(); ++col) {
        const std::string& value = values[col];
        const Column& column = table.getColumns()[col];
        
        // Skip empty values
        if (value.empty()) {
            // Add a default value based on type
            switch (column.getType()) {
                case DataType::INT:
                    table.appendIntValue(col, 0);
                    break;
                case DataType::FLOAT:
                    table.appendFloatValue(col, 0.0f);
                    break;
                case DataType::DOUBLE:
                    table.appendDoubleValue(col, 0.0);
                    break;
                case DataType::VARCHAR:
                case DataType::STRING:
                    table.appendStringValue(col, "");
                    break;
                case DataType::BOOL:
                    table.appendBoolValue(col, false);
                    break;
                case DataType::DATE:
                    table.appendStringValue(col, ""); // Store dates as strings
                    break;
                case DataType::DATETIME:
                    table.appendStringValue(col, ""); // Empty datetime
                    break;    
            }
            continue;
        }
        
        // Otherwise, parse the value according to column type
        try {
            switch (column.getType()) {
                case DataType::INT:
                    table.appendIntValue(col, std::stoi(value));
                    break;
                case DataType::FLOAT:
                    table.appendFloatValue(col, std::stof(value));
                    break;
                case DataType::DOUBLE:
                    if (value.find('.') != std::string::npos) {
                        table.appendDoubleValue(col, std::stod(value));
                    } else {
                        // Integer value in a DOUBLE column
                        table.appendDoubleValue(col, static_cast<double>(std::stoi(value)));
                    }
                    break;
                case DataType::VARCHAR:
                case DataType::STRING:
                    table.appendStringValue(col, value);
                    break;
                case DataType::BOOL:
                    table.appendBoolValue(col, value == "true" || value == "1");
                    break;
                case DataType::DATE:
                    // Store dates as strings for now
                    table.appendStringValue(col, value);
                    break;
                case DataType::DATETIME:
                    // For DateTime, validate the format before adding
                    if (Table::isValidDateTime(value)) {
                        table.appendStringValue(col, value);
                    } else {
                        // Try to convert to proper format if possible
                        // For now, store as-is
                        table.appendStringValue(col, value);
                    }
                    break;
            }
        } catch (const std::exception& e) {
            // If conversion fails, use a default value
            std::cerr << "Warning: Failed to convert value '" << value 
                      << "' to type " << static_cast<int>(column.getType()) 
                      << " for column " << column.getName() << ": " << e.what() << std::endl;
            
            // Add a default value
            switch (column.getType()) {
                case DataType::INT:
                    table.appendIntValue(col, 0);
                    break;
                case DataType::FLOAT:
                    table.appendFloatValue(col, 0.0f);
                    break;
                case DataType::DOUBLE:
                    table.appendDoubleValue(col, 0.0);
                    break;
                case DataType::VARCHAR:
                case DataType::STRING:
                case DataType::DATE:
                    table.appendStringValue(col, value); // Keep as string
                    break;
                case DataType::BOOL:
                    table.appendBoolValue(col, false);
                    break;
            }
        }
    }
}

std::string CSVProcessor::escapeCSVString(const std::string& str) {
    std::string result;
    for (char c : str) {
        if (c == '"') {
            result += "\"\""; // Escape quotes by doubling them
        } else {
            result += c;
        }
    }
    return result;
}

std::string CSVProcessor::formatHeaderWithTypeAnnotations(const std::string& columnName, DataType type, bool isPrimary) {
    std::string header = columnName;
    
    // Add type annotation
    switch (type) {
        case DataType::INT:
        case DataType::FLOAT:
        case DataType::DOUBLE:
            header += " (N)";
            break;
        case DataType::VARCHAR:
        case DataType::STRING:
            header += " (T)";
            break;
        case DataType::BOOL:
            header += " (B)";
            break;
        case DataType::DATETIME:
            header += " (D)";
            break;
        default:
            header += " (T)"; // Default to text
    }
    
    // Add primary key annotation if needed
    if (isPrimary) {
        header += " (P)";
    }
    
    return header;
}

void StorageManager::parseTableConstraints(const std::string& tableName, const std::vector<std::string>& headers) {
    // Regular expressions for detecting primary and foreign keys
    std::regex primaryKeyRegex("(.+) \\(P\\)"); // Column (P) format
    std::regex foreignKeyRegex("(.+)_id");      // TableName_id format
    
    TableMetadata& metadata = m_tableMetadata[tableName];
    
    for (const auto& header : headers) {
        std::string cleanHeader = header;
        
        // If header contains multiple annotations, strip them for regex matching
        size_t firstParen = cleanHeader.find(" (");
        if (firstParen != std::string::npos) {
            cleanHeader = cleanHeader.substr(0, firstParen);
        }
        
        std::smatch match;
        
        // Check if it's a primary key (original header should contain (P))
        if (header.find(" (P)") != std::string::npos) {
            metadata.primaryKeys.push_back(cleanHeader);
            std::cout << "Found primary key in " << tableName << ": " << cleanHeader << std::endl;
        }
        
        // Check if it's a foreign key (ends with _id but is not a primary key)
        if (cleanHeader.length() > 3 && 
            cleanHeader.substr(cleanHeader.length() - 3) == "_id" && 
            std::find(metadata.primaryKeys.begin(), metadata.primaryKeys.end(), cleanHeader) == metadata.primaryKeys.end()) {
            
            // Extract the target table name (everything before _id)
            std::string targetTable = cleanHeader.substr(0, cleanHeader.length() - 3);
            
            // If this looks like a reference to another table
            if (!targetTable.empty() && targetTable != tableName) {
                ForeignKeyConstraint fk;
                fk.sourceColumn = cleanHeader;
                fk.targetTable = targetTable;
                fk.targetColumn = targetTable + "_id"; // Assume target is the primary key with same name
                
                metadata.foreignKeys.push_back(fk);
                std::cout << "Found foreign key in " << tableName << ": " << cleanHeader 
                          << " references " << targetTable << "." << fk.targetColumn << std::endl;
            }
        }
    }
}

Table& StorageManager::getTable(const std::string& tableName) {
    if (!hasTable(tableName)) {
        // Try to load it first
        try {
            loadTableFromCSV(tableName);
        } catch (const std::exception& e) {
            throw std::runtime_error("Table not found: " + tableName);
        }
    }
    
    return m_tables.at(tableName);
}

bool StorageManager::hasTable(const std::string& tableName) const {
    return m_tables.find(tableName) != m_tables.end();
}

void StorageManager::registerTable(const std::string& tableName, const Table& table) {
    m_tables[tableName] = table;
    
    // Setup basic metadata
    TableMetadata metadata;
    metadata.name = tableName;
    metadata.filePath = m_dataDirectory + "/outputs/csv/" + tableName + ".csv";
    m_tableMetadata[tableName] = metadata;
}

std::vector<std::string> StorageManager::getTableNames() const {
    std::vector<std::string> names;
    names.reserve(m_tables.size());
    
    for (const auto& [name, _] : m_tables) {
        names.push_back(name);
    }
    
    return names;
}

const TableMetadata& StorageManager::getTableMetadata(const std::string& tableName) const {
    auto it = m_tableMetadata.find(tableName);
    if (it == m_tableMetadata.end()) {
        throw std::runtime_error("No metadata found for table: " + tableName);
    }
    return it->second;
}

} // namespace GPUDBMS