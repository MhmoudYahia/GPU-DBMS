<<<<<<< Updated upstream
#ifndef TABLE_HPP
#define TABLE_HPP

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <stdexcept>

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
        STRING // Added STRING type based on usage
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

        const T &getValue(size_t index) const
        {
            if (index >= m_data.size())
            {
                throw std::out_of_range("Index out of range");
            }
            return m_data.at(index);
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
         * @brief Add a new column to the table
         *
         * @param column The column definition to add
         */
        void addColumn(const Column &column);

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

    private:
        std::vector<Column> m_columns;
        std::vector<std::unique_ptr<ColumnData>> m_columnData;
        std::unordered_map<std::string, size_t> m_columnNameToIndex;
    };

} // namespace GPUDBMS

=======
#ifndef TABLE_HPP
#define TABLE_HPP

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <stdexcept>

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
        STRING // Added STRING type based on usage
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

        const T &getValue(size_t index) const
        {
            if (index >= m_data.size())
            {
                throw std::out_of_range("Index out of range");
            }
            return m_data.at(index);
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
         * @brief Add a new column to the table
         *
         * @param column The column definition to add
         */
        void addColumn(const Column &column);

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

        /**
         * @brief Create a new table with the same schema but no data
         *
         * @return Table An empty table with the same schema
         */
        Table createEmptyWithSameSchema() const;

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

    private:
        std::vector<Column> m_columns;
        std::vector<std::unique_ptr<ColumnData>> m_columnData;
        std::unordered_map<std::string, size_t> m_columnNameToIndex;
    };

} // namespace GPUDBMS

>>>>>>> Stashed changes
#endif // TABLE_HPP