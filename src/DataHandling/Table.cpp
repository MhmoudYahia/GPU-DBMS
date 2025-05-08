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
            colInfo.data = const_cast<int*>(col.getData().data());
            colInfo.count = col.size();
            colInfo.stride = sizeof(int); // Assuming int data is stored in a contiguous block
            break;
        }
        case GPUDBMS::DataType::FLOAT:
        {
            auto &col = static_cast<const GPUDBMS::ColumnDataImpl<float> &>(cd);
            colInfo.data = const_cast<float*>(col.getData().data());
            colInfo.count = col.size();
            colInfo.stride = sizeof(float); // Assuming float data is stored in a contiguous block
            break;
        }
        case GPUDBMS::DataType::DOUBLE:
        {
            auto &col = static_cast<const GPUDBMS::ColumnDataImpl<double> &>(cd);
            colInfo.data = const_cast<double*>(col.getData().data());
            colInfo.count = col.size();
            colInfo.stride = sizeof(double); // Assuming double data is stored in a contiguous block
            break;
        }
        case GPUDBMS::DataType::VARCHAR:
        case GPUDBMS::DataType::STRING:
        {
            auto &col = static_cast<const GPUDBMS::ColumnDataImpl<std::string> &>(cd);
            const std::vector<std::string> &strVec = col.getData();

            // Allocate a single contiguous buffer for all strings
            const size_t maxStrLen = 256; // For VARCHAR/STRING
            char *h_contiguousBuffer = new char[strVec.size() * maxStrLen];

            // Copy strings to contiguous buffer
            for (size_t i = 0; i < strVec.size(); i++)
            {
                strncpy(h_contiguousBuffer + (i * maxStrLen),
                        strVec[i].c_str(),
                        maxStrLen - 1);
                h_contiguousBuffer[(i * maxStrLen) + maxStrLen - 1] = '\0';
            }

            colInfo.data = h_contiguousBuffer;
            colInfo.count = strVec.size();
            colInfo.stride = maxStrLen; // Add this to your ColumnInfo struct
            break;
        }
        case GPUDBMS::DataType::BOOL:
        {
            auto &col = static_cast<const GPUDBMS::ColumnDataImpl<bool> &>(cd);
            // Create a copy of the data since std::vector<bool> doesn't provide direct pointer access
            static std::vector<char> boolBuffer;
            boolBuffer.clear();
            const auto& boolData = col.getData();
            boolBuffer.reserve(boolData.size());
            for (bool val : boolData) {
                boolBuffer.push_back(val ? 1 : 0);
            }
            colInfo.data = boolBuffer.data();
            colInfo.count = col.size();
            colInfo.stride = sizeof(char); // Assuming bool data is stored as char (0 or 1)
            break;
        }

        case GPUDBMS::DataType::DATE:
        case GPUDBMS::DataType::DATETIME:
        {
            auto &col = static_cast<const GPUDBMS::ColumnDataImpl<std::string> &>(cd);
            const std::vector<std::string> &strVec = col.getData();

            // Allocate a single contiguous buffer for all strings
            const size_t maxStrLen = 20; // For DATE/DATETIME
            char *h_contiguousBuffer = new char[strVec.size() * maxStrLen];

            // Copy strings to contiguous buffer
            for (size_t i = 0; i < strVec.size(); i++)
            {
                strncpy(h_contiguousBuffer + (i * maxStrLen),
                        strVec[i].c_str(),
                        maxStrLen - 1);
                h_contiguousBuffer[(i * maxStrLen) + maxStrLen - 1] = '\0';
            }

            colInfo.data = h_contiguousBuffer;
            colInfo.count = strVec.size();
            colInfo.stride = maxStrLen;
            break;
        }
           
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
                  
        // Accept either DATETIME/DATE or VARCHAR/STRING for datetime values
        if (column.getType() != DataType::DATETIME && column.getType() != DataType::DATE) {
            // For non-datetime columns, still try to get the value if it's a string type
            if (column.getType() == DataType::VARCHAR || column.getType() == DataType::STRING) {
                const auto &typedColumn = static_cast<const ColumnDataImpl<std::string> &>(column);
                std::string value = typedColumn.getValue(rowIndex);
                
                // Basic validation to ensure it looks like a date
                if (isValidDateTime(value)) {
                    return value;
                }
                
                // std::cout << "Warning: Value '" << value << "' doesn't appear to be a valid datetime" << std::endl;
                return value; // Return it anyway
            }
            
            throw std::runtime_error("Column is not of type DATETIME or DATE and not a string type");
        }
        
        const auto &typedColumn = static_cast<const ColumnDataImpl<std::string> &>(column);
        return typedColumn.getValue(rowIndex);
    }

    // Add a method to append DateTime values
    void Table::appendDateTimeValue(size_t columnIndex, const std::string &value)
    {
        ColumnData &column = getColumnData(columnIndex);
        std::cout << "Column type: " << static_cast<int>(column.getType()) << std::endl;
        if (column.getType() != DataType::DATETIME)
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

    Table Table::createSlicedEmptyWithSameSchema(const std::vector<std::string> &projectColumns) const
    {
        Table newTable;

        // Copy schema and track which columns to include
        std::vector<bool> includeColumn(m_columns.size(), false);
        for (size_t i = 0; i < m_columns.size(); ++i)
        {
            const auto &column = m_columns[i];
            if (std::find(projectColumns.begin(), projectColumns.end(), column.getName()) != projectColumns.end())
            {
                newTable.m_columns.push_back(column);
                newTable.m_columnNameToIndex[column.getName()] = newTable.m_columns.size() - 1;
                includeColumn[i] = true;
            }
        }

        // Create empty column data with same types
        for (size_t i = 0; i < m_columnData.size(); ++i)
        {
            if (includeColumn[i])
            {
                newTable.m_columnData.push_back(m_columnData[i]->createEmpty());
            }
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