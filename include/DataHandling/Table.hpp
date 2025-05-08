#ifndef TABLE_HPP
#define TABLE_HPP

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include <chrono>

namespace GPUDBMS
{

    /**
     * @enum DataType
     * @brief Supported data types for table columns
     */
    enum class DataType
    {
        INT,
        FLOAT,
        DOUBLE,
        VARCHAR,
        BOOL,
        DATE,
        STRING,   // Added STRING type based on usage
        DATETIME, // New type for yyyy-MM-dd HH:mm:ss format
    };

    struct ColumnInfoGPU
    {
        DataType type;
        std::string name;
        size_t count;
        size_t stride; // Add this for string types
        void *data;
    };

    /**
     * @class Column
     * @brief Represents a column in a database table
     */
    class Column
    {
    public:
        /**
         * @brief Construct a new Column
         *
         * @param name The column name
         * @param type The data type of the column
         */
        Column(const std::string &name, DataType type);

        /**
         * @brief Get the name of the column
         *
         * @return const std::string& The column name
         */
        const std::string &getName() const;

        void setName(const std::string& newName) {
            m_name = newName;
        }
        
        /**
         * @brief Get the data type of the column
         *
         * @return DataType The column's data type
         */
        DataType getType() const;

    private:
        std::string m_name;
        DataType m_type;
    };

    /**
     * @class ColumnData
     * @brief Abstract base class for column data storage
     */
    class ColumnData
    {
    public:
        virtual ~ColumnData() = default;
        virtual size_t size() const = 0;
        virtual DataType getType() const = 0;
        virtual std::unique_ptr<ColumnData> clone() const = 0;
        virtual std::unique_ptr<ColumnData> createEmpty() const = 0;
        virtual void appendFromRow(const ColumnData &source, int rowIndex) = 0;
        virtual void reserve(size_t count) = 0;
    };

    /**
     * @class ColumnDataImpl
     * @brief Type-specific implementation of column data storage
     */
    template <typename T>
    class ColumnDataImpl : public ColumnData
    {
    public:
        ColumnDataImpl() = default;

        ColumnDataImpl(const std::vector<T> &data)
            : m_data(data) {}

        void append(const T &value)
        {
            m_data.push_back(value);
        }

        T getValue(size_t index) const
        {
            if (index >= m_data.size())
            {
                throw std::out_of_range("Index out of range");
            }
            return m_data[index];
        }

        const std::vector<T> &getData() const
        {
            return m_data;
        }

        std::vector<T> &getData()
        {
            return m_data;
        }

        size_t size() const override
        {
            return m_data.size();
        }

        DataType getType() const override;

        std::unique_ptr<ColumnData> clone() const override
        {
            return std::make_unique<ColumnDataImpl<T>>(m_data);
        }

        std::unique_ptr<ColumnData> createEmpty() const override
        {
            return std::make_unique<ColumnDataImpl<T>>();
        }

        void appendFromRow(const ColumnData &source, int rowIndex)
        {
            const auto &typedSource = static_cast<const ColumnDataImpl<T> &>(source);
            if (rowIndex < 0 || rowIndex >= typedSource.size())
            {
                throw std::out_of_range("Row index out of range");
            }
            append(typedSource.getValue(rowIndex));
        }


        void reserve(size_t count) override
        {
            m_data.reserve(count);
        }

    private:
        std::vector<T> m_data;
    };

    /**
     * @class Table
     * @brief Represents a database table with columns and rows
     */
    class Table
    {
    public:
        /**
         * @brief Construct an empty table
         */
        Table();

        /**
         * @brief Construct a table with the given schema
         *
         * @param columns The columns that define the table schema
         */
        Table(const std::vector<Column> &columns);

        /**
         * @brief Copy constructor
         */
        Table(const Table &other);

        /**
         * @brief Move constructor
         */
        Table(Table &&other) noexcept;

        /**
         * @brief Assignment operator
         */
        Table &operator=(const Table &other);

        /**
         * @brief Move assignment operator
         */
        Table &operator=(Table &&other) noexcept;

        /**
         * @brief Get a datetime value from the table
         *
         * @param columnIndex The column index
         * @param rowIndex The row index
         * @return std::string The datetime value at the specified position
         */
        std::string getDateTimeValue(size_t columnIndex, size_t rowIndex) const;

        /**
         * @brief Append a datetime value to a column
         *
         * @param columnIndex The column index
         * @param value The datetime value to append
         */
        void appendDateTimeValue(size_t columnIndex, const std::string &value);

        /**
         * @brief get table name
         */
        std::string getTableName() const;

        /**
         * @brief Validate a datetime string format
         *
         * @param dateTime The datetime string to validate
         * @return bool True if the string is a valid datetime format
         */
        static bool isValidDateTime(const std::string &dateTime);

        /**
         * @brief Add a new column to the table
         *
         * @param column The column definition to add
         */
        void addColumn(const Column &column);

        /**
         * @brief Set column data using a template for different data types
         *
         * @tparam T The data type of the column
         * @param columnIndex The index of the column
         * @param data The data to set for the column
         */
        template <typename T>
        void setColumnData(size_t columnIndex, std::vector<T> data)
        {
            if (columnIndex >= m_columnData.size())
            {
                throw std::out_of_range("Column index out of range");
            }

            auto columnType = m_columns[columnIndex].getType();
            auto inputType = ColumnDataImpl<T>().getType();

            // Special handling for char vectors (typically used for string data)
            if constexpr (std::is_same<T, char>::value)
            {
                if (columnType == GPUDBMS::DataType::STRING ||
                    columnType == GPUDBMS::DataType::VARCHAR ||
                    columnType == GPUDBMS::DataType::DATE ||
                    columnType == GPUDBMS::DataType::DATETIME)
                {
                    // Handle as string data
                    auto &columnData = dynamic_cast<ColumnDataImpl<std::string> &>(*m_columnData[columnIndex]);
                    std::string strData(data.begin(), data.end());
                    columnData.getData() = std::vector<std::string>{strData};
                    return;
                }
            }

            // Normal type checking for non-char cases
            if (columnType != inputType)
            {
                throw std::invalid_argument("Data type mismatch for column " +
                                            m_columns[columnIndex].getName() +
                                            " expected " + std::to_string(static_cast<int>(columnType)) +
                                            " but got " + std::to_string(static_cast<int>(inputType)));
            }

            // Standard handling for matching types
            try
            {
                auto &columnData = dynamic_cast<ColumnDataImpl<T> &>(*m_columnData[columnIndex]);
                columnData.getData() = data;
            }
            catch (const std::bad_cast &e)
            {
                throw std::invalid_argument("Type conversion failed for column " +
                                            m_columns[columnIndex].getName() +
                                            ": " + e.what());
            }
        }
        /**
         * @brief Get all columns in the table
         *
         * @return const std::vector<Column>& The table columns
         */
        const std::vector<Column> &getColumns() const;

        /**
         * @brief Get the number of rows in the table
         *
         * @return size_t Row count
         */
        size_t getRowCount() const;

        /**
         * @brief Get the number of columns in the table
         *
         * @return size_t Column count
         */
        size_t getColumnCount() const;

        /**
         * @brief Get the index of a column by name
         *
         * @param columnName The name of the column
         * @return int The index of the column, or -1 if not found
         */
        int getColumnIndex(const std::string &columnName) const;

        /**
         * @brief Get access to a column's data by index
         *
         * @param columnIndex The index of the column
         * @return ColumnData& Reference to the column data
         */
        ColumnData &getColumnData(size_t columnIndex);

        /**
         * @brief Get const access to a column's data by index
         *
         * @param columnIndex The index of the column
         * @return const ColumnData& Const reference to the column data
         */
        const ColumnData &getColumnData(size_t columnIndex) const;

        /**
         * @brief Get access to a column's data by name
         *
         * @param columnName The name of the column
         * @return ColumnData& Reference to the column data
         */
        ColumnData &getColumnData(const std::string &columnName);

        /**
         * @brief Get const access to a column's data by name
         *
         * @param columnName The name of the column
         * @return const ColumnData& Const reference to the column data
         */
        const ColumnData &getColumnData(const std::string &columnName) const;

        std::vector<DataType> getColumnsType();

        const std::vector<DataType> getColumnsType() const;

        /**
         * @brief Get the data type of a column by index
         *
         * @param columnIndex The index of the column
         * @return DataType The data type of the column
         */
        DataType getColumnType(size_t columnIndex) const;

        /**
         * @brief Create a new table with the same schema but no data
         *
         * @return Table An empty table with the same schema
         */
        Table createEmptyWithSameSchema() const;

        Table createSlicedEmptyWithSameSchema(const std::vector<std::string> &projectColumns) const;

        /**
         * @brief Get an integer value from the table
         *
         * @param columnIndex The column index
         * @param rowIndex The row index
         * @return int The integer value at the specified position
         */
        int getIntValue(size_t columnIndex, size_t rowIndex) const;

        /**
         * @brief Get a float value from the table
         *
         * @param columnIndex The column index
         * @param rowIndex The row index
         * @return float The float value at the specified position
         */
        float getFloatValue(size_t columnIndex, size_t rowIndex) const;

        /**
         * @brief Get a string value from the table
         *
         * @param columnIndex The column index
         * @param rowIndex The row index
         * @return std::string The string value at the specified position
         */
        std::string getStringValue(size_t columnIndex, size_t rowIndex) const;

        /**
         * @brief Get a double value from the table
         *
         * @param columnIndex The column index
         * @param rowIndex The row index
         * @return double The double value at the specified position
         */
        double getDoubleValue(size_t columnIndex, size_t rowIndex) const;

        /**
         * @brief Get a boolean value from the table
         *
         * @param columnIndex The column index
         * @param rowIndex The row index
         * @return bool The boolean value at the specified position
         */
        bool getBoolValue(size_t columnIndex, size_t rowIndex) const;

        /**
         * @brief Append an integer value to a column
         *
         * @param columnIndex The column index
         * @param value The integer value to append
         */
        void appendIntValue(size_t columnIndex, int value);

        /**
         * @brief Append a float value to a column
         *
         * @param columnIndex The column index
         * @param value The float value to append
         */
        void appendFloatValue(size_t columnIndex, float value);

        // Get the name of a column by index
        std::string getColumnName(size_t index) const;

        DataType getColumnType(const std::string &columnName) const;
        /**
         * @brief Append a string value to a column
         *
         * @param columnIndex The column index
         * @param value The string value to append
         */
        void appendStringValue(size_t columnIndex, const std::string &value);

        /**
         * @brief Append a double value to a column
         *
         * @param columnIndex The column index
         * @param value The double value to append
         */
        void appendDoubleValue(size_t columnIndex, double value);
        /**
         * @brief Append a boolean value to a column
         *
         * @param columnIndex The column index
         * @param value The boolean value to append
         */
        void appendBoolValue(size_t columnIndex, bool value);

        /**
         * @brief Finalize a row after appending values to all columns
         */
        void finalizeRow();

        /**
         * @brief Create a new table containing only the specified rows
         *
         * @param rowIndices The indices of rows to include in the new table
         * @return Table A new table with the selected rows
         */
        Table getSlicedTable(const std::vector<int> &rowIndices) const;

        ColumnInfoGPU getColumnInfoGPU(const std::string &columnName) const;

        void reserve(size_t rowCount)
        {
            for (auto &columnData : m_columnData)
            {
                // Dispatch to the correct type-specific reserve implementation
                columnData->reserve(rowCount);
            }
        }

    private:
        std::vector<Column>
            m_columns;
        std::vector<std::unique_ptr<ColumnData>> m_columnData;
        std::unordered_map<std::string, size_t> m_columnNameToIndex;
    };

} // namespace GPUDBMS

#endif // TABLE_HPP