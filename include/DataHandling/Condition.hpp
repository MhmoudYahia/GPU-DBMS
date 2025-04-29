#ifndef CONDITION_HPP
#define CONDITION_HPP

#include <string>
#include <functional>
#include <vector>
#include <memory>
#include "../DataHandling/Table.hpp"

namespace GPUDBMS
{

    /**
     * @enum ComparisonOperator
     * @brief Enumeration of comparison operators for conditions
     */
    enum class ComparisonOperator
    {
        EQUAL,
        NOT_EQUAL,
        LESS_THAN,
        LESS_EQUAL,
        GREATER_THAN,
        GREATER_EQUAL,
        LIKE,
        IN
    };

    /**
     * @enum LogicalOperator
     * @brief Enumeration of logical operators for combining conditions
     */
    enum class LogicalOperator
    {
        AND,
        OR,
        NOT
    };

    /**
     * @class Condition
     * @brief Base class representing a condition for filtering data
     */
    class Condition
    {
    public:
        virtual ~Condition() = default;

        Condition();
        Condition(const Condition &other);                // Copy constructor
        Condition(Condition &&other) noexcept;            // Move constructor
        Condition &operator=(Condition &&other) noexcept; // Move assignment operator
        Condition &operator=(const Condition &other);     // Copy assignment operator

        /**
         * @brief Evaluate the condition for a given row
         *
         * @param row The row data to evaluate against
         * @param columnIndices The indices of columns in the row
         * @return bool Whether the condition is satisfied
         */
        virtual bool evaluate(const std::vector<DataType> &colsType, const std::vector<std::string> &row, std::unordered_map<std::string, int> columnNameToIndex) const = 0;

        /**
         * @brief Get the CUDA compatible condition string for GPU execution
         *
         * @return std::string A string representation of the condition for CUDA kernels
         */
        virtual std::string getCUDACondition() const = 0;

        /**
         * @brief Clone the condition
         *
         * @return std::unique_ptr<Condition> A copy of this condition
         */
        virtual std::unique_ptr<Condition> clone() const = 0;
    };

    /**
     * @class ComparisonCondition
     * @brief Represents a comparison between a column and a value
     */
    class ComparisonCondition : public Condition
    {
    public:
        /**
         * @brief Construct a comparison condition
         *
         * @param columnName The name of the column to compare
         * @param op The comparison operator
         * @param value The value to compare against
         */
        ComparisonCondition(const std::string &columnName, ComparisonOperator op, const std::string &value);
        ComparisonCondition(const ComparisonCondition &other);
        ComparisonCondition &operator=(const ComparisonCondition &other);
        ComparisonCondition(ComparisonCondition &&other) noexcept;
        ComparisonCondition &operator=(ComparisonCondition &&other) noexcept;
        ~ComparisonCondition() override = default;

        bool evaluate(const std::vector<DataType> &colsType, const std::vector<std::string> &row, std::unordered_map<std::string, int> columnNameToIndex) const override;
        std::string getCUDACondition() const override;
        std::unique_ptr<Condition> clone() const override;

        /**
         * @brief Get the column name involved in this condition
         *
         * @return const std::string& The column name
         */
        const std::string &getColumnName() const;

        /**
         * @brief Get the comparison operator
         *
         * @return ComparisonOperator The operator used for comparison
         */
        ComparisonOperator getOperator() const;

        /**
         * @brief Get the value being compared against
         *
         * @return const std::string& The comparison value
         */
        const std::string &getValue() const;

    private:
        std::string m_columnName;
        ComparisonOperator m_operator;
        std::string m_value;
    };

    /**
     * @class LogicalCondition
     * @brief Represents a logical combination of multiple conditions
     */
    class LogicalCondition : public Condition
    {
    public:
        /**
         * @brief Construct a logical condition with two operands
         *
         * @param left The left operand
         * @param op The logical operator
         * @param right The right operand (only for AND/OR)
         */
        LogicalCondition(std::unique_ptr<Condition> left, LogicalOperator op, std::unique_ptr<Condition> right = nullptr);

        bool evaluate(const std::vector<DataType> &colsType, const std::vector<std::string> &row, std::unordered_map<std::string, int> columnNameToIndex) const override;
        std::string getCUDACondition() const override;
        std::unique_ptr<Condition> clone() const override;

    private:
        std::unique_ptr<Condition> m_left;
        LogicalOperator m_operator;
        std::unique_ptr<Condition> m_right;
    };

    /**
     * @class ConditionBuilder
     * @brief Helper class for building complex conditions
     */
    class ConditionBuilder
    {
    public:
        static std::unique_ptr<Condition> equals(const std::string &columnName, const std::string &value);
        static std::unique_ptr<Condition> notEquals(const std::string &columnName, const std::string &value);
        static std::unique_ptr<Condition> lessThan(const std::string &columnName, const std::string &value);
        static std::unique_ptr<Condition> lessEqual(const std::string &columnName, const std::string &value);
        static std::unique_ptr<Condition> greaterThan(const std::string &columnName, const std::string &value);
        static std::unique_ptr<Condition> greaterEqual(const std::string &columnName, const std::string &value);
        static std::unique_ptr<Condition> like(const std::string &columnName, const std::string &pattern);

        static std::unique_ptr<Condition> And(std::unique_ptr<Condition> left, std::unique_ptr<Condition> right);
        static std::unique_ptr<Condition> Or(std::unique_ptr<Condition> left, std::unique_ptr<Condition> right);
        static std::unique_ptr<Condition> Not(std::unique_ptr<Condition> condition);
    };

} // namespace GPUDBMS

#endif // CONDITION_HPP