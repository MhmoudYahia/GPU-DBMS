#include "DataHandling/Table.hpp"
#include "Utilities/ErrorHandling.hpp"
#include "Utilities/Logger.hpp"
#include <algorithm>
#include <sstream>

namespace SQLQueryProcessor {

Table::Table(const std::string& tableName)
    : name(tableName), rowCount(0) {
    Logger::debug("Created table: " + tableName);
}

Table::Table(const Table& other)
    : name(other.name + "_copy"),
      columnNames(other.columnNames),
      primaryKeyColumns(other.primaryKeyColumns),
      foreignKeyColumns(other.foreignKeyColumns),
      tableData(other.tableData),
      rowCount(other.rowCount) {
    Logger::debug("Created table copy: " + name);
}

std::shared_ptr<Table> Table::createSimilarTable(const std::string& newTableName) const {
    auto newTable = std::make_shared<Table>(newTableName);
    
    // Copy the schema (column names and key info) but not the data
    for (size_t i = 0; i < columnNames.size(); i++) {
        newTable->addColumn(
            columnNames[i], 
            std::find(primaryKeyColumns.begin(), primaryKeyColumns.end(), i) != primaryKeyColumns.end(),
            std::find(foreignKeyColumns.begin(), foreignKeyColumns.end(), i) != foreignKeyColumns.end()
        );
    }
    
    return newTable;
}

std::string Table::getName() const {
    return name;
}

void Table::setName(const std::string& newName) {
    name = newName;
}

size_t Table::getRowCount() const {
    return rowCount;
}

size_t Table::getColumnCount() const {
    return columnNames.size();
}

void Table::addColumn(const std::string& columnName, bool isPrimaryKey, bool isForeignKey) {
    // Check if column already exists
    auto it = std::find(columnNames.begin(), columnNames.end(), columnName);
    if (it != columnNames.end()) {
        throw DataHandlingException("Column '" + columnName + "' already exists in table '" + name + "'");
    }
    
    // Add column
    columnNames.push_back(columnName);
    size_t columnIndex = columnNames.size() - 1;
    
    // Set key attributes
    if (isPrimaryKey) {
        primaryKeyColumns.push_back(columnIndex);
    }
    
    if (isForeignKey) {
        foreignKeyColumns.push_back(columnIndex);
    }
    
    // Add empty values for existing rows
    for (auto& row : tableData) {
        row.push_back("");
    }
    
    Logger::debug("Added column '" + columnName + "' to table '" + name + "'");
}

void Table::renameColumn(const std::string& oldName, const std::string& newName) {
    // Find the column index
    int columnIndex = getColumnIndex(oldName);
    if (columnIndex < 0) {
        throw DataHandlingException("Column '" + oldName + "' not found in table '" + name + "'");
    }
    
    // Check if new name already exists
    auto it = std::find(columnNames.begin(), columnNames.end(), newName);
    if (it != columnNames.end()) {
        throw DataHandlingException("Column '" + newName + "' already exists in table '" + name + "'");
    }
    
    // Rename column
    columnNames[columnIndex] = newName;
    
    Logger::debug("Renamed column from '" + oldName + "' to '" + newName + "' in table '" + name + "'");
}

void Table::addRow(const std::vector<std::string>& rowData) {
    // Validate row size
    if (rowData.size() != columnNames.size()) {
        throw DataHandlingException("Row data has " + std::to_string(rowData.size()) + 
                                  " columns, but table '" + name + "' has " + 
                                  std::to_string(columnNames.size()) + " columns");
    }
    
    // Add the row
    tableData.push_back(rowData);
    rowCount++;
}

bool Table::updateRow(size_t rowIndex, const std::vector<std::string>& rowData) {
    // Validate row index
    if (rowIndex >= rowCount) {
        Logger::warning("Row index " + std::to_string(rowIndex) + " out of bounds in table '" + name + "'");
        return false;
    }
    
    // Validate row size
    if (rowData.size() != columnNames.size()) {
        Logger::warning("Row data has " + std::to_string(rowData.size()) + 
                       " columns, but table '" + name + "' has " + 
                       std::to_string(columnNames.size()) + " columns");
        return false;
    }
    
    // Update the row
    tableData[rowIndex] = rowData;
    return true;
}

bool Table::deleteRow(size_t rowIndex) {
    // Validate row index
    if (rowIndex >= rowCount) {
        Logger::warning("Row index " + std::to_string(rowIndex) + " out of bounds in table '" + name + "'");
        return false;
    }
    
    // Delete the row
    tableData.erase(tableData.begin() + rowIndex);
    rowCount--;
    return true;
}

const std::vector<std::string>& Table::getColumnNames() const {
    return columnNames;
}

int Table::getColumnIndex(const std::string& columnName) const {
    // Find the column index
    for (size_t i = 0; i < columnNames.size(); i++) {
        if (columnNames[i] == columnName) {
            return static_cast<int>(i);
        }
    }
    
    // Handle table-qualified column names (table.column)
    size_t dotPos = columnName.find('.');
    if (dotPos != std::string::npos) {
        std::string tableName = columnName.substr(0, dotPos);
        std::string colName = columnName.substr(dotPos + 1);
        
        // Check if the table name matches this table
        if (tableName == name) {
            for (size_t i = 0; i < columnNames.size(); i++) {
                if (columnNames[i] == colName) {
                    return static_cast<int>(i);
                }
            }
        } else {
            // Also check if the column name includes this table's name already
            for (size_t i = 0; i < columnNames.size(); i++) {
                if (columnNames[i] == columnName) {
                    return static_cast<int>(i);
                }
            }
        }
    }
    
    return -1;  // Column not found
}

std::string Table::getValue(size_t rowIndex, size_t columnIndex) const {
    // Validate indices
    if (rowIndex >= rowCount) {
        throw DataHandlingException("Row index " + std::to_string(rowIndex) + 
                                  " out of bounds in table '" + name + "'");
    }
    
    if (columnIndex >= columnNames.size()) {
        throw DataHandlingException("Column index " + std::to_string(columnIndex) + 
                                  " out of bounds in table '" + name + "'");
    }
    
    // Return the value
    return tableData[rowIndex][columnIndex];
}

std::string Table::getValue(size_t rowIndex, const std::string& columnName) const {
    int columnIndex = getColumnIndex(columnName);
    if (columnIndex < 0) {
        throw DataHandlingException("Column '" + columnName + "' not found in table '" + name + "'");
    }
    
    return getValue(rowIndex, static_cast<size_t>(columnIndex));
}

bool Table::setValue(size_t rowIndex, size_t columnIndex, const std::string& value) {
    // Validate indices
    if (rowIndex >= rowCount) {
        Logger::warning("Row index " + std::to_string(rowIndex) + 
                       " out of bounds in table '" + name + "'");
        return false;
    }
    
    if (columnIndex >= columnNames.size()) {
        Logger::warning("Column index " + std::to_string(columnIndex) + 
                       " out of bounds in table '" + name + "'");
        return false;
    }
    
    // Update the value
    tableData[rowIndex][columnIndex] = value;
    return true;
}

bool Table::setValue(size_t rowIndex, const std::string& columnName, const std::string& value) {
    int columnIndex = getColumnIndex(columnName);
    if (columnIndex < 0) {
        Logger::warning("Column '" + columnName + "' not found in table '" + name + "'");
        return false;
    }
    
    return setValue(rowIndex, static_cast<size_t>(columnIndex), value);
}

bool Table::isColumnPrimaryKey(size_t columnIndex) const {
    return std::find(primaryKeyColumns.begin(), primaryKeyColumns.end(), columnIndex) != primaryKeyColumns.end();
}

bool Table::isColumnForeignKey(size_t columnIndex) const {
    return std::find(foreignKeyColumns.begin(), foreignKeyColumns.end(), columnIndex) != foreignKeyColumns.end();
}

void Table::setPrimaryKey(size_t columnIndex, bool isPrimary) {
    // Validate column index
    if (columnIndex >= columnNames.size()) {
        throw DataHandlingException("Column index " + std::to_string(columnIndex) + 
                                  " out of bounds in table '" + name + "'");
    }
    
    // Update primary key status
    auto it = std::find(primaryKeyColumns.begin(), primaryKeyColumns.end(), columnIndex);
    
    if (isPrimary && it == primaryKeyColumns.end()) {
        primaryKeyColumns.push_back(columnIndex);
        Logger::debug("Column '" + columnNames[columnIndex] + "' set as primary key in table '" + name + "'");
    } else if (!isPrimary && it != primaryKeyColumns.end()) {
        primaryKeyColumns.erase(it);
        Logger::debug("Column '" + columnNames[columnIndex] + "' removed as primary key in table '" + name + "'");
    }
}

void Table::setForeignKey(size_t columnIndex, bool isForeign) {
    // Validate column index
    if (columnIndex >= columnNames.size()) {
        throw DataHandlingException("Column index " + std::to_string(columnIndex) + 
                                  " out of bounds in table '" + name + "'");
    }
    
    // Update foreign key status
    auto it = std::find(foreignKeyColumns.begin(), foreignKeyColumns.end(), columnIndex);
    
    if (isForeign && it == foreignKeyColumns.end()) {
        foreignKeyColumns.push_back(columnIndex);
        Logger::debug("Column '" + columnNames[columnIndex] + "' set as foreign key in table '" + name + "'");
    } else if (!isForeign && it != foreignKeyColumns.end()) {
        foreignKeyColumns.erase(it);
        Logger::debug("Column '" + columnNames[columnIndex] + "' removed as foreign key in table '" + name + "'");
    }
}

std::vector<size_t> Table::getPrimaryKeyColumns() const {
    return primaryKeyColumns;
}

std::vector<size_t> Table::getForeignKeyColumns() const {
    return foreignKeyColumns;
}

std::vector<std::string> Table::getRow(size_t rowIndex) const {
    // Validate row index
    if (rowIndex >= rowCount) {
        throw DataHandlingException("Row index " + std::to_string(rowIndex) + 
                                  " out of bounds in table '" + name + "'");
    }
    
    return tableData[rowIndex];
}

const std::vector<std::vector<std::string>>& Table::getData() const {
    return tableData;
}

void Table::clear() {
    tableData.clear();
    rowCount = 0;
    Logger::debug("Cleared all data from table '" + name + "'");
}

std::string Table::toString() const {
    std::stringstream ss;
    
    // Table name
    ss << "Table: " << name << std::endl;
    
    // Column headers with key indicators
    for (size_t i = 0; i < columnNames.size(); i++) {
        ss << columnNames[i];
        
        if (isColumnPrimaryKey(i)) {
            ss << " (PK)";
        }
        
        if (isColumnForeignKey(i)) {
            ss << " (FK)";
        }
        
        if (i < columnNames.size() - 1) {
            ss << "\t";
        }
    }
    ss << std::endl;
    
    // Separator line
    for (size_t i = 0; i < columnNames.size(); i++) {
        ss << std::string(columnNames[i].length() + 
                        (isColumnPrimaryKey(i) ? 5 : 0) + 
                        (isColumnForeignKey(i) ? 5 : 0), '-');
        
        if (i < columnNames.size() - 1) {
            ss << "\t";
        }
    }
    ss << std::endl;
    
    // Table data
    for (size_t i = 0; i < rowCount; i++) {
        for (size_t j = 0; j < columnNames.size(); j++) {
            ss << tableData[i][j];
            
            if (j < columnNames.size() - 1) {
                ss << "\t";
            }
        }
        
        if (i < rowCount - 1) {
            ss << std::endl;
        }
    }
    
    return ss.str();
}

bool Table::hasColumn(const std::string& columnName) const {
    return getColumnIndex(columnName) >= 0;
}

} // namespace SQLQueryProcessor