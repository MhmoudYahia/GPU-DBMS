#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <stdexcept>

namespace SQLQueryProcessor
{

    class Table
    {
    private:
        std::string name;
        std::vector<std::string> columnNames;
        std::unordered_map<std::string, int> columnIndices;
        std::vector<std::vector<std::string>> data;
        std::vector<bool> isPrimaryKey;
        std::vector<bool> isForeignKey;
        std::unordered_map<std::string, std::string> foreignKeyReferences; // Column name -> referenced table

    public:
        Table(const std::string &tableName);
        ~Table() = default;

        // Column management
        void addColumn(const std::string &columnName, bool isPrimary = false, bool isForeign = false);
        void setForeignKeyReference(const std::string &columnName, const std::string &referencedTable);

        // Data management
        void addRow(const std::vector<std::string> &rowData);
        void clear();

        // Accessors
        const std::string &getName() const;
        const std::vector<std::string> &getColumnNames() const;
        int getColumnIndex(const std::string &columnName) const;
        bool isColumnPrimaryKey(int columnIndex) const;
        bool isColumnForeignKey(int columnIndex) const;
        std::string getForeignKeyReference(const std::string &columnName) const;

        // Data access methods
        size_t getRowCount() const;
        size_t getColumnCount() const;
        const std::string &getValue(size_t rowIndex, size_t columnIndex) const;
        const std::vector<std::vector<std::string>> &getData() const;
        std::vector<std::vector<std::string>> &getDataMutable();

        // Create a new table with the same schema
        std::shared_ptr<Table> createSimilarTable(const std::string &newName) const;
    };

} // namespace SQLQueryProcessor