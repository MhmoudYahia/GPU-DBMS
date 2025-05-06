#include "../../include/DataHandling/Table.hpp"
#include <algorithm>
#include <regex>

namespace GPUDBMS
{

    // Column implementation
    Column::Column(const std::string &name, DataType type)
        : m_name(name), m_type(type) {}

    const std::string &Column::getName() const
    {
        return m_name;
    }

    int Table::getIntValue(size_t columnIndex, size_t rowIndex) const
    {
        const ColumnData &column = getColumnData(columnIndex);
        if (column.getType() != DataType::INT)
        {
            throw std::runtime_error("Column is not of type INT");
        }
        const auto &typedColumn = static_cast<const ColumnDataImpl<int> &>(column);
        if (rowIndex >= typedColumn.size())
        {
            throw std::out_of_range("Row index out of range");
        }
        return typedColumn.getValue(rowIndex);
    }

    float Table::getFloatValue(size_t columnIndex, size_t rowIndex) const
    {
        const ColumnData &column = getColumnData(columnIndex);
        if (column.getType() != DataType::FLOAT)
        {
            throw std::runtime_error("Column is not of type FLOAT");
        }
        const auto &typedColumn = static_cast<const ColumnDataImpl<float> &>(column);
        return typedColumn.getValue(rowIndex);
    }

    std::string Table::getStringValue(size_t columnIndex, size_t rowIndex) const
    {
        const ColumnData &column = getColumnData(columnIndex);
        if (column.getType() != DataType::VARCHAR)
        {
            throw std::runtime_error("Column is not of type VARCHAR");
        }
        const auto &typedColumn = static_cast<const ColumnDataImpl<std::string> &>(column);
        return typedColumn.getValue(rowIndex);
    }

    double Table::getDoubleValue(size_t columnIndex, size_t rowIndex) const
    {
        const ColumnData &column = getColumnData(columnIndex);
        if (column.getType() != DataType::DOUBLE)
        {
            throw std::runtime_error("Column is not of type DOUBLE");
        }
        const auto &typedColumn = static_cast<const ColumnDataImpl<double> &>(column);
        return typedColumn.getValue(rowIndex);
    }

    bool Table::getBoolValue(size_t columnIndex, size_t rowIndex) const
    {
        const ColumnData &column = getColumnData(columnIndex);
        if (column.getType() != DataType::BOOL)
        {
            throw std::runtime_error("Column is not of type BOOL");
        }
        const auto &typedColumn = static_cast<const ColumnDataImpl<bool> &>(column);
        if (rowIndex >= typedColumn.size())
        {
            throw std::out_of_range("Row index out of range");
        }
        return typedColumn.getValue(rowIndex);
    }

    void Table::appendIntValue(size_t columnIndex, int value)
    {
        ColumnData &column = getColumnData(columnIndex);
        if (column.getType() != DataType::INT)
        {
            throw std::runtime_error("Column is not of type INT");
        }
        auto &typedColumn = static_cast<ColumnDataImpl<int> &>(column);
        typedColumn.append(value);
    }

    void Table::appendFloatValue(size_t columnIndex, float value)
    {
        ColumnData &column = getColumnData(columnIndex);
        if (column.getType() != DataType::FLOAT)
        {
            throw std::runtime_error("Column is not of type FLOAT");
        }
        auto &typedColumn = static_cast<ColumnDataImpl<float> &>(column);
        typedColumn.append(value);
    }

    void Table::appendStringValue(size_t columnIndex, const std::string &value)
    {
        ColumnData &column = getColumnData(columnIndex);
        if (column.getType() != DataType::VARCHAR)
        {
            throw std::runtime_error("Column is not of type VARCHAR");
        }
        auto &typedColumn = static_cast<ColumnDataImpl<std::string> &>(column);
        typedColumn.append(value);
    }

    void Table::appendDoubleValue(size_t columnIndex, double value)
    {
        ColumnData &column = getColumnData(columnIndex);
        if (column.getType() != DataType::DOUBLE)
        {
            throw std::runtime_error("Column is not of type DOUBLE");
        }
        auto &typedColumn = static_cast<ColumnDataImpl<double> &>(column);
        typedColumn.append(value);
    }

    void Table::appendBoolValue(size_t columnIndex, bool value)
    {
        ColumnData &column = getColumnData(columnIndex);
        if (column.getType() != DataType::BOOL)
        {
            throw std::runtime_error("Column is not of type BOOL");
        }
        auto &typedColumn = static_cast<ColumnDataImpl<bool> &>(column);
        typedColumn.append(value);
    }

    void Table::finalizeRow()
    {

        // Ensure all columns have the same number of rows
        if (m_columnData.empty())
        {
            return;
        }

        size_t rowCount = m_columnData[0]->size();
        for (size_t i = 1; i < m_columnData.size(); ++i)
        {
            if (m_columnData[i]->size() != rowCount)
            {
                throw std::runtime_error("Column " + m_columns[i].getName() +
                                         " has inconsistent row count");
            }
        }
    }

    ColumnInfoGPU Table::getColumnInfoGPU(const std::string &columnName) const
    {
        ColumnInfoGPU colInfo;

        auto it = m_columnNameToIndex.find(columnName);
        colInfo.type = m_columns[it->second].getType();
        colInfo.name = columnName;

        const GPUDBMS::ColumnData &cd = getColumnData(it->second);

        switch (colInfo.type)
        {
        case GPUDBMS::DataType::INT:
        {
            auto &col = static_cast<const GPUDBMS::ColumnDataImpl<int> &>(cd);
            colInfo.data = col.getData().data();
            break;
        }
        case GPUDBMS::DataType::FLOAT:
        {
            auto &col = static_cast<const GPUDBMS::ColumnDataImpl<float> &>(cd);
            colInfo.data = col.getData().data();
            break;
        }
        case GPUDBMS::DataType::DOUBLE:
        {
            auto &col = static_cast<const GPUDBMS::ColumnDataImpl<double> &>(cd);
            colInfo.data = col.getData().data();
            break;
        }
            // case GPUDBMS::DataType::BOOL: {
            //     auto &col = static_cast<const GPUDBMS::ColumnDataImpl<bool>&>(cd);
            //     colInfo.data = col.getData().data();
            //     break;
            // }
            // Add cases for other data types as needed
        }
        return colInfo;
    }

    DataType Column::getType() const
    {
        return m_type;
    }

    // Template specializations for getType
    template <>
    DataType ColumnDataImpl<int>::getType() const
    {
        return DataType::INT;
    }

    template <>
    DataType ColumnDataImpl<float>::getType() const
    {
        return DataType::FLOAT;
    }

    template <>
    DataType ColumnDataImpl<double>::getType() const
    {
        return DataType::DOUBLE;
    }

    template <>
    DataType ColumnDataImpl<std::string>::getType() const
    {
        return DataType::VARCHAR;
    }

    template <>
    DataType ColumnDataImpl<bool>::getType() const
    {
        return DataType::BOOL;
    }

    // Table implementation
    Table::Table() {}

    Table::Table(const std::vector<Column> &columns)
    {
        for (const auto &column : columns)
        {
            addColumn(column);
        }
    }

    Table::Table(const Table &other)
        : m_columns(other.m_columns), m_columnNameToIndex(other.m_columnNameToIndex)
    {
        // Deep copy of column data
        for (const auto &columnData : other.m_columnData)
        {
            m_columnData.push_back(columnData->clone());
        }
    }

    Table::Table(Table &&other) noexcept
        : m_columns(std::move(other.m_columns)),
          m_columnData(std::move(other.m_columnData)),
          m_columnNameToIndex(std::move(other.m_columnNameToIndex)) {}

    Table &Table::operator=(const Table &other)
    {
        if (this != &other)
        {
            m_columns = other.m_columns;
            m_columnNameToIndex = other.m_columnNameToIndex;

            // Deep copy of column data
            m_columnData.clear();
            for (const auto &columnData : other.m_columnData)
            {
                m_columnData.push_back(columnData->clone());
            }
        }
        return *this;
    }

    Table &Table::operator=(Table &&other) noexcept
    {
        if (this != &other)
        {
            m_columns = std::move(other.m_columns);
            m_columnData = std::move(other.m_columnData);
            m_columnNameToIndex = std::move(other.m_columnNameToIndex);
        }
        return *this;
    }

    void Table::addColumn(const Column &column)
    {
        const std::string &name = column.getName();

        // Check if column with this name already exists
        if (m_columnNameToIndex.find(name) != m_columnNameToIndex.end())
        {
            throw std::runtime_error("Column with name '" + name + "' already exists in table");
        }

        m_columns.push_back(column);
        m_columnNameToIndex[name] = m_columns.size() - 1;

        // Create appropriate column data storage based on type
        switch (column.getType())
        {
        case DataType::INT:
            m_columnData.push_back(std::make_unique<ColumnDataImpl<int>>());
            break;
        case DataType::FLOAT:
            m_columnData.push_back(std::make_unique<ColumnDataImpl<float>>());
            break;
        case DataType::DOUBLE:
            m_columnData.push_back(std::make_unique<ColumnDataImpl<double>>());
            break;
        case DataType::VARCHAR:
            m_columnData.push_back(std::make_unique<ColumnDataImpl<std::string>>());
            break;
        case DataType::BOOL:
            m_columnData.push_back(std::make_unique<ColumnDataImpl<bool>>());
            break;

        case DataType::DATE:
        case DataType::DATETIME: // Add datetime support using string storage
            m_columnData.push_back(std::make_unique<ColumnDataImpl<std::string>>());
            break;
        default:
            throw std::runtime_error("Unsupported data type for column: " + name);
        }
    }

    template <typename T>
    void Table::setColumnData(size_t columnIndex, std::vector<T> data)
    {
        if (columnIndex >= m_columnData.size())
        {
            throw std::out_of_range("Column index out of range");
        }

        auto columnType = m_columns[columnIndex].getType();
        if (columnType != ColumnDataImpl<T>().getType())
        {
            throw std::invalid_argument("Data type mismatch for column");
        }

        auto &columnData = dynamic_cast<ColumnDataImpl<T> &>(*m_columnData[columnIndex]);
        columnData.getData() = data;
    }

    const std::vector<Column> &Table::getColumns() const
    {
        return m_columns;
    }

    size_t Table::getRowCount() const
    {
        if (m_columnData.empty())
        {
            return 0;
        }
        return m_columnData[0]->size();
    }

    size_t Table::getColumnCount() const
    {
        return m_columns.size();
    }

    int Table::getColumnIndex(const std::string &columnName) const
    {
        auto it = m_columnNameToIndex.find(columnName);
        if (it != m_columnNameToIndex.end())
        {
            return static_cast<int>(it->second);
        }
        return -1;
    }

    ColumnData &Table::getColumnData(size_t columnIndex)
    {
        if (columnIndex >= m_columnData.size())
        {
            throw std::out_of_range("Column index out of range");
        }
        return *m_columnData[columnIndex];
    }

    std::vector<DataType> Table::getColumnsType()
    {
        std::vector<DataType> types;
        for (const auto &column : m_columns)
        {
            types.push_back(column.getType());
        }
        return types;
    }

    const std::vector<DataType> Table::getColumnsType() const
    {
        std::vector<DataType> types;
        for (const auto &column : m_columns)
        {
            types.push_back(column.getType());
        }
        return types;
    }

    DataType Table::getColumnType(size_t columnIndex) const
    {
        if (columnIndex >= m_columns.size())
        {
            throw std::out_of_range("Column index out of range");
        }
        return m_columns[columnIndex].getType();
    }

    const ColumnData &Table::getColumnData(size_t columnIndex) const
    {
        if (columnIndex >= m_columnData.size())
        {
            throw std::out_of_range("Column index out of range");
        }
        return *m_columnData[columnIndex];
    }

    DataType Table::getColumnType(const std::string &columnName) const
    {
        int index = getColumnIndex(columnName);
        if (index == -1)
        {
            throw std::runtime_error("Column not found: " + columnName);
        }
        return m_columns[index].getType();
    }

    // Add a method to get DateTime value (essentially the same as getString but with validation)
    std::string Table::getDateTimeValue(size_t columnIndex, size_t rowIndex) const
    {
        const ColumnData &column = getColumnData(columnIndex);
        if (column.getType() != DataType::DATETIME && column.getType() != DataType::DATE)
        {
            throw std::runtime_error("Column is not of type DATETIME or DATE");
        }
        const auto &typedColumn = static_cast<const ColumnDataImpl<std::string> &>(column);
        return typedColumn.getValue(rowIndex);
    }

    // Add a method to append DateTime values
    void Table::appendDateTimeValue(size_t columnIndex, const std::string &value)
    {
        ColumnData &column = getColumnData(columnIndex);
        if (column.getType() != DataType::DATETIME && column.getType() != DataType::DATE)
        {
            throw std::runtime_error("Column is not of type DATETIME or DATE");
        }

        // Validate datetime format (optional but recommended)
        if (!isValidDateTime(value))
        {
            throw std::runtime_error("Invalid DateTime format. Expected yyyy-MM-dd HH:mm:ss but got: " + value);
        }

        auto &typedColumn = static_cast<ColumnDataImpl<std::string> &>(column);
        typedColumn.append(value);
    }

    // Helper method to validate DateTime format
    bool Table::isValidDateTime(const std::string &dateTime)
    {
        // Basic validation for yyyy-MM-dd HH:mm:ss format
        std::regex dateTimePattern("^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}$");
        return std::regex_match(dateTime, dateTimePattern);
    }

    ColumnData &Table::getColumnData(const std::string &columnName)
    {
        int index = getColumnIndex(columnName);
        if (index == -1)
        {
            throw std::runtime_error("Column not found: " + columnName);
        }
        return *m_columnData[index];
    }

    const ColumnData &Table::getColumnData(const std::string &columnName) const
    {
        int index = getColumnIndex(columnName);
        if (index == -1)
        {
            throw std::runtime_error("Column not found: " + columnName);
        }
        return *m_columnData[index];
    }

    Table Table::createEmptyWithSameSchema() const
    {
        Table newTable;

        // Copy schema
        newTable.m_columns = m_columns;
        newTable.m_columnNameToIndex = m_columnNameToIndex;

        // Create empty column data with same types
        for (const auto &columnData : m_columnData)
        {
            newTable.m_columnData.push_back(columnData->createEmpty());
        }

        return newTable;
    }

    // Get the name of a column by index
    std::string Table::getColumnName(size_t index) const
    {
        if (index >= m_columns.size())
        {
            throw std::out_of_range("Column index out of range");
        }
        return m_columns[index].getName();
    }

    Table Table::getSlicedTable(const std::vector<int> &rowIndices) const
    {
        Table resultTable = createEmptyWithSameSchema();

        const size_t colCount = m_columns.size();
        std::vector<const GPUDBMS::ColumnData *> sourceColumns(colCount);
        std::vector<GPUDBMS::ColumnData *> resultColumns(colCount);

        for (size_t col = 0; col < colCount; ++col)
        {
            sourceColumns[col] = &getColumnData(col);
            resultColumns[col] = &resultTable.getColumnData(col);
        }

        for (const auto &rowIndex : rowIndices)
        {
            for (size_t col = 0; col < colCount; ++col)
            {
                resultColumns[col]->appendFromRow(*sourceColumns[col], rowIndex);
            }
        }

        resultTable.finalizeRow();
        return resultTable;
    }

    template <typename T>
    void ColumnDataImpl<T>::appendFromRow(const ColumnData &source, int rowIndex)
    {
        const auto &typedSource = static_cast<const ColumnDataImpl<T> &>(source);
        if (rowIndex < 0 || rowIndex >= typedSource.size())
        {
            throw std::out_of_range("Row index out of range");
        }
        append(typedSource.getValue(rowIndex));
    }

} // namespace GPUDBMS