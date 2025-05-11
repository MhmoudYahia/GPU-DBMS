#include "../../include/DataHandling/Condition.hpp"
#include "../../include/DataHandling/ConditionGPU.cuh"
#include <iomanip> // For std::get_time
#include <ctime>
#include <algorithm>
#include "../../include/DataHandling/Table.hpp" // Include the header defining DataType
#include <regex>
#include <sstream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace GPUDBMS
{
    // Condition implementation
    Condition::Condition()
    {
    }
    Condition::Condition(const Condition &other)
    {
        // Copy constructor
    }
    Condition &Condition::operator=(const Condition &other)
    {
        if (this != &other)
        {
            // Assignment operator
        }
        return *this;
    }
    Condition::Condition(Condition &&other) noexcept
    {
        // Move constructor
    }
    Condition &Condition::operator=(Condition &&other) noexcept
    {
        if (this != &other)
        {
            // Move assignment operator
        }
        return *this;
    }

    // ComparisonCondition implementation
    ComparisonCondition::ComparisonCondition(const std::string &columnName, ComparisonOperator op, const std::string &value)
        : m_columnName(columnName), m_operator(op), m_value(value)
    {
    }

    ComparisonCondition::ComparisonCondition(const ComparisonCondition &other)
        : m_columnName(other.m_columnName), m_operator(other.m_operator), m_value(other.m_value)
    {
    }
    ComparisonCondition &ComparisonCondition::operator=(const ComparisonCondition &other)
    {
        if (this != &other)
        {
            m_columnName = other.m_columnName;
            m_operator = other.m_operator;
            m_value = other.m_value;
        }
        return *this;
    }

    ComparisonCondition::ComparisonCondition(ComparisonCondition &&other) noexcept
        : m_columnName(std::move(other.m_columnName)), m_operator(other.m_operator), m_value(std::move(other.m_value))
    {
    }
    ComparisonCondition &ComparisonCondition::operator=(ComparisonCondition &&other) noexcept
    {
        if (this != &other)
        {
            m_columnName = std::move(other.m_columnName);
            m_operator = other.m_operator;
            m_value = std::move(other.m_value);
        }
        return *this;
    }

    bool ComparisonCondition::evaluate(const std::vector<DataType> &colsType, const std::vector<std::string> &row, std::unordered_map<std::string, int> columnNameToIndex) const
    {
        // Find the index of the column in columnIndices
        auto it = columnNameToIndex.find(m_columnName);

        if (it == columnNameToIndex.end())
        {
            return false; // Column not found
        }

        int index = it->second;
        if (index >= row.size() || index >= colsType.size())
        {
            return false; // Invalid index
        }

        const std::string &cellValue = row[index];
        const DataType &colType = colsType[index];

        switch (m_operator)
        {
        case ComparisonOperator::EQUAL:
            if (colType == DataType::INT)
                return std::stoi(cellValue) == std::stoi(m_value);
            else if (colType == DataType::FLOAT || colType == DataType::DOUBLE)
                return std::stof(cellValue) == std::stof(m_value);
            else if (colType == DataType::DATETIME || colType == DataType::DATE)
                return compareDateTime(cellValue, m_value) == 0;
            else
                return cellValue == m_value;
        case ComparisonOperator::NOT_EQUAL:
            if (colType == DataType::INT)
                return std::stoi(cellValue) != std::stoi(m_value);
            else if (colType == DataType::FLOAT || colType == DataType::DOUBLE)
                return std::stof(cellValue) != std::stof(m_value);
            else if (colType == DataType::DATETIME || colType == DataType::DATE)
                return compareDateTime(cellValue, m_value) != 0;
            else
                return cellValue != m_value;
        case ComparisonOperator::LESS_THAN:
            if (colType == DataType::INT)
                return std::stoi(cellValue) < std::stoi(m_value);
            else if (colType == DataType::FLOAT || colType == DataType::DOUBLE)
                return std::stof(cellValue) < std::stof(m_value);
            else if (colType == DataType::DATETIME || colType == DataType::DATE)
                return compareDateTime(cellValue, m_value) < 0;
            else
                return cellValue < m_value;
        case ComparisonOperator::LESS_EQUAL:
            if (colType == DataType::INT)
                return std::stoi(cellValue) <= std::stoi(m_value);
            else if (colType == DataType::FLOAT || colType == DataType::DOUBLE)
                return std::stof(cellValue) <= std::stof(m_value);

            else if (colType == DataType::DATETIME || colType == DataType::DATE)
                return compareDateTime(cellValue, m_value) <= 0;
            else
                return cellValue <= m_value;
        case ComparisonOperator::GREATER_THAN:
            if (colType == DataType::INT)
                return std::stoi(cellValue) > std::stoi(m_value);
            else if (colType == DataType::FLOAT || colType == DataType::DOUBLE)
                return std::stof(cellValue) > std::stof(m_value);

            else if (colType == DataType::DATETIME || colType == DataType::DATE)
                return compareDateTime(cellValue, m_value) > 0;
            else
                return cellValue > m_value;
        case ComparisonOperator::GREATER_EQUAL:
            if (colType == DataType::INT)
                return std::stoi(cellValue) >= std::stoi(m_value);
            else if (colType == DataType::FLOAT || colType == DataType::DOUBLE)
                return std::stof(cellValue) >= std::stof(m_value);
            else if (colType == DataType::DATETIME || colType == DataType::DATE)
                return compareDateTime(cellValue, m_value) >= 0;
            else
                return cellValue >= m_value;
        case ComparisonOperator::LIKE:
        {
            // Simple wildcard pattern matching (% matches any sequence)
            std::string pattern = m_value;
            // Replace % with regex .*
            std::string regexPattern;
            for (char c : pattern)
            {
                if (c == '%')
                {
                    regexPattern += ".*";
                }
                else
                {
                    regexPattern += c;
                }
            }
            std::regex regex(regexPattern);
            return std::regex_match(cellValue, regex);
        }
        case ComparisonOperator::IN:
            // Simple implementation - assumes value is comma-separated list
            return m_value.find(cellValue) != std::string::npos;
        default:
            return false;
        }
    }

    // Add implementation for ComparisonCondition::toString()
    std::string ComparisonCondition::toString() const
    {
        std::string opStr;
        switch (m_operator)
        {
        case ComparisonOperator::EQUAL:
            opStr = " = ";
            break;
        case ComparisonOperator::NOT_EQUAL:
            opStr = " != ";
            break;
        case ComparisonOperator::LESS_THAN:
            opStr = " < ";
            break;
        case ComparisonOperator::LESS_EQUAL:
            opStr = " <= ";
            break;
        case ComparisonOperator::GREATER_THAN:
            opStr = " > ";
            break;
        case ComparisonOperator::GREATER_EQUAL:
            opStr = " >= ";
            break;
        case ComparisonOperator::LIKE:
            opStr = " LIKE ";
            break;
        case ComparisonOperator::IN:
            opStr = " IN ";
            break;
        }

        return m_columnName + opStr + m_value;
    }

    // Add implementation for LogicalCondition::toString()
    std::string LogicalCondition::toString() const
    {
        std::string opStr;
        switch (m_operator)
        {
        case LogicalOperator::AND:
            opStr = " AND ";
            break;
        case LogicalOperator::OR:
            opStr = " OR ";
            break;
        case LogicalOperator::NOT:
            opStr = "NOT ";
            break;
        case LogicalOperator::NONE:
            opStr = "";
            break;
        }

        if (m_operator == LogicalOperator::NOT)
        {
            return opStr + "(" + m_left->toString() + ")";
        }
        else if (m_right) // AND or OR
        {
            return "(" + m_left->toString() + ")" + opStr + "(" + m_right->toString() + ")";
        }
        else // Just one operand
        {
            return m_left->toString();
        }
    }

    int ComparisonCondition::compareDateTime(const std::string &dateTime1, const std::string &dateTime2) const
    {
        // If either is empty, handle that case
        if (dateTime1.empty() || dateTime2.empty())
        {
            return dateTime1.empty() ? -1 : 1;
        }

        try
        {
            // Parse dates - handle common ISO formats
            std::tm tm1 = {}, tm2 = {};
            std::istringstream ss1(dateTime1), ss2(dateTime2);

            // Try to parse as YYYY-MM-DD HH:MM:SS format
            if (dateTime1.length() > 10)
            { // Full datetime
                ss1 >> std::get_time(&tm1, "%Y-%m-%d %H:%M:%S");
            }
            else
            { // Just date
                ss1 >> std::get_time(&tm1, "%Y-%m-%d");
            }

            if (dateTime2.length() > 10)
            { // Full datetime
                ss2 >> std::get_time(&tm2, "%Y-%m-%d %H:%M:%S");
            }
            else
            { // Just date
                ss2 >> std::get_time(&tm2, "%Y-%m-%d");
            }

            // Check if parsing was successful
            if (ss1.fail() || ss2.fail())
            {
                // Fall back to string comparison if parsing fails
                return dateTime1.compare(dateTime2);
            }

            // Convert to time_t for easy comparison
            std::time_t time1 = std::mktime(&tm1);
            std::time_t time2 = std::mktime(&tm2);

            if (time1 == time2)
                return 0;
            return (time1 < time2) ? -1 : 1;
        }
        catch (const std::exception &)
        {
            // If any exception occurs, fall back to string comparison
            return dateTime1.compare(dateTime2);
        }
    }

    // bool *ComparisonCondition::evaluateGPU(
    //     const std::vector<DataType> &colsType,
    //     const std::vector<std::string> &row,
    //     std::unordered_map<std::string, int> columnNameToIndex) const
    // {
    //     return launchFilterKernel(m_columnName, m_operator, m_value, colsType, row, columnNameToIndex);
    // }

    std::string ComparisonCondition::getCUDACondition() const
    {
        std::string opStr;
        switch (m_operator)
        {
        case ComparisonOperator::EQUAL:
            opStr = "==";
            break;
        case ComparisonOperator::NOT_EQUAL:
            opStr = "!=";
            break;
        case ComparisonOperator::LESS_THAN:
            opStr = "<";
            break;
        case ComparisonOperator::LESS_EQUAL:
            opStr = "<=";
            break;
        case ComparisonOperator::GREATER_THAN:
            opStr = ">";
            break;
        case ComparisonOperator::GREATER_EQUAL:
            opStr = ">=";
            break;
        case ComparisonOperator::LIKE:
            opStr = "LIKE";
            break;
        case ComparisonOperator::IN:
            opStr = "IN";
            break;
        }

        return "(" + m_columnName + " " + opStr + " " + m_value + ")";
    }

    std::unique_ptr<Condition> ComparisonCondition::clone() const
    {
        return std::make_unique<ComparisonCondition>(m_columnName, m_operator, m_value);
    }

    const std::string &ComparisonCondition::getColumnName() const
    {
        return m_columnName;
    }

    ComparisonOperator ComparisonCondition::getOperator() const
    {
        return m_operator;
    }

    const std::string &ComparisonCondition::getValue() const
    {
        return m_value;
    }

    // LogicalCondition implementation
    LogicalCondition::LogicalCondition(std::unique_ptr<Condition> left, LogicalOperator op, std::unique_ptr<Condition> right)
        : m_left(std::move(left)), m_operator(op), m_right(std::move(right))
    {
    }

    bool LogicalCondition::evaluate(const std::vector<DataType> &colsType, const std::vector<std::string> &row, std::unordered_map<std::string, int> columnNameToIndex) const
    {
        switch (m_operator)
        {
        case LogicalOperator::AND:
            return m_left->evaluate(colsType, row, columnNameToIndex) && m_right->evaluate(colsType, row, columnNameToIndex);
        case LogicalOperator::OR:
            return m_left->evaluate(colsType, row, columnNameToIndex) || m_right->evaluate(colsType, row, columnNameToIndex);
        case LogicalOperator::NOT:
            return !m_left->evaluate(colsType, row, columnNameToIndex);
        default:
            return false;
        }
    }

    // bool *LogicalCondition::evaluateGPU(const std::vector<DataType> &colsType, const std::vector<std::string> &row, std::unordered_map<std::string, int> columnNameToIndex) const
    // {
    //     // Prepare GPU data
    //     int numRows = row.size();
    //     int **intCols = new int *[numRows];
    //     float **floatCols = new float *[numRows];
    //     bool **boolCols = new bool *[numRows];
    //     char *stringBuffer = new char[256 * numRows]; // Assuming max string length of 256
    //     int *stringOffsets = new int[numRows];
    //     bool *outputFlags = new bool[numRows];

    //     std::fill(outputFlags, outputFlags + numRows, false);
    //     return outputFlags;
    // }

    std::string LogicalCondition::getCUDACondition() const
    {
        std::string opStr;
        switch (m_operator)
        {
        case LogicalOperator::AND:
            opStr = "&&";
            break;
        case LogicalOperator::OR:
            opStr = "||";
            break;
        case LogicalOperator::NOT:
            opStr = "!";
            break;
        }

        if (m_operator == LogicalOperator::NOT)
        {
            return "(" + opStr + m_left->getCUDACondition() + ")";
        }
        else
        {
            return "(" + m_left->getCUDACondition() + " " + opStr + " " + m_right->getCUDACondition() + ")";
        }
    }

    std::unique_ptr<Condition> LogicalCondition::clone() const
    {
        if (m_operator == LogicalOperator::NOT)
        {
            return std::make_unique<LogicalCondition>(m_left->clone(), m_operator);
        }
        else
        {
            return std::make_unique<LogicalCondition>(m_left->clone(), m_operator, m_right->clone());
        }
    }

    // ConditionBuilder implementation
    std::unique_ptr<Condition> ConditionBuilder::equals(const std::string &columnName, const std::string &value)
    {
        return std::make_unique<ComparisonCondition>(columnName, ComparisonOperator::EQUAL, value);
    }

    std::unique_ptr<Condition> ConditionBuilder::notEquals(const std::string &columnName, const std::string &value)
    {
        return std::make_unique<ComparisonCondition>(columnName, ComparisonOperator::NOT_EQUAL, value);
    }

    std::unique_ptr<Condition> ConditionBuilder::lessThan(const std::string &columnName, const std::string &value)
    {
        return std::make_unique<ComparisonCondition>(columnName, ComparisonOperator::LESS_THAN, value);
    }

    std::unique_ptr<Condition> ConditionBuilder::lessEqual(const std::string &columnName, const std::string &value)
    {
        return std::make_unique<ComparisonCondition>(columnName, ComparisonOperator::LESS_EQUAL, value);
    }

    std::unique_ptr<Condition> ConditionBuilder::greaterThan(const std::string &columnName, const std::string &value)
    {
        return std::make_unique<ComparisonCondition>(columnName, ComparisonOperator::GREATER_THAN, value);
    }

    std::unique_ptr<Condition> ConditionBuilder::greaterEqual(const std::string &columnName, const std::string &value)
    {
        return std::make_unique<ComparisonCondition>(columnName, ComparisonOperator::GREATER_EQUAL, value);
    }

    std::unique_ptr<Condition> ConditionBuilder::like(const std::string &columnName, const std::string &pattern)
    {
        return std::make_unique<ComparisonCondition>(columnName, ComparisonOperator::LIKE, pattern);
    }

    std::unique_ptr<Condition> ConditionBuilder::And(std::unique_ptr<Condition> left, std::unique_ptr<Condition> right)
    {
        return std::make_unique<LogicalCondition>(std::move(left), LogicalOperator::AND, std::move(right));
    }

    std::unique_ptr<Condition> ConditionBuilder::Or(std::unique_ptr<Condition> left, std::unique_ptr<Condition> right)
    {
        return std::make_unique<LogicalCondition>(std::move(left), LogicalOperator::OR, std::move(right));
    }

    std::unique_ptr<Condition> ConditionBuilder::Not(std::unique_ptr<Condition> condition)
    {
        return std::make_unique<LogicalCondition>(std::move(condition), LogicalOperator::NOT);
    }

    // Add implementation for ColumnComparisonCondition

    // ColumnComparisonCondition implementation
    ColumnComparisonCondition::ColumnComparisonCondition(const std::string &leftColumn, ComparisonOperator op, const std::string &rightColumn)
        : m_leftColumn(leftColumn), m_operator(op), m_rightColumn(rightColumn)
    {
    }

    ColumnComparisonCondition::ColumnComparisonCondition(const ColumnComparisonCondition &other)
        : m_leftColumn(other.m_leftColumn), m_operator(other.m_operator), m_rightColumn(other.m_rightColumn)
    {
    }

    ColumnComparisonCondition &ColumnComparisonCondition::operator=(const ColumnComparisonCondition &other)
    {
        if (this != &other)
        {
            m_leftColumn = other.m_leftColumn;
            m_operator = other.m_operator;
            m_rightColumn = other.m_rightColumn;
        }
        return *this;
    }

    ColumnComparisonCondition::ColumnComparisonCondition(ColumnComparisonCondition &&other) noexcept
        : m_leftColumn(std::move(other.m_leftColumn)), m_operator(other.m_operator), m_rightColumn(std::move(other.m_rightColumn))
    {
    }

    ColumnComparisonCondition &ColumnComparisonCondition::operator=(ColumnComparisonCondition &&other) noexcept
    {
        if (this != &other)
        {
            m_leftColumn = std::move(other.m_leftColumn);
            m_operator = other.m_operator;
            m_rightColumn = std::move(other.m_rightColumn);
        }
        return *this;
    }

    bool ColumnComparisonCondition::evaluate(const std::vector<DataType> &colsType, const std::vector<std::string> &row,
                                             std::unordered_map<std::string, int> columnNameToIndex) const
    {
        // Find indices of left and right columns
        auto leftIt = columnNameToIndex.find(m_leftColumn);
        auto rightIt = columnNameToIndex.find("right_" + m_rightColumn);

        if (leftIt == columnNameToIndex.end() || rightIt == columnNameToIndex.end())
        {
            return false; // Column not found
        }

        int leftIndex = leftIt->second;
        int rightIndex = rightIt->second;

        if (leftIndex >= row.size() || rightIndex >= row.size() ||
            leftIndex >= colsType.size() || rightIndex >= colsType.size())
        {
            return false; // Invalid index
        }

        const std::string &leftValue = row[leftIndex];
        const std::string &rightValue = row[rightIndex];
        const DataType &leftType = colsType[leftIndex];
        const DataType &rightType = colsType[rightIndex];

        // Compare based on data types
        switch (m_operator)
        {
        case ComparisonOperator::EQUAL:
            if (leftType == DataType::INT && rightType == DataType::INT)
                return std::stoi(leftValue) == std::stoi(rightValue);
            else if ((leftType == DataType::FLOAT || leftType == DataType::DOUBLE) &&
                     (rightType == DataType::FLOAT || rightType == DataType::DOUBLE))
                return std::stof(leftValue) == std::stof(rightValue);
            else if ((leftType == DataType::DATETIME || leftType == DataType::DATE) &&
                     (rightType == DataType::DATETIME || rightType == DataType::DATE))
            {
                ComparisonCondition tempCond("", ComparisonOperator::EQUAL, "");
                return tempCond.compareDateTime(leftValue, rightValue) == 0;
            }
            else
                return leftValue == rightValue;
        case ComparisonOperator::NOT_EQUAL:
            if (leftType == DataType::INT && rightType == DataType::INT)
                return std::stoi(leftValue) != std::stoi(rightValue);
            else if ((leftType == DataType::FLOAT || leftType == DataType::DOUBLE) &&
                     (rightType == DataType::FLOAT || rightType == DataType::DOUBLE))
                return std::stof(leftValue) != std::stof(rightValue);
            else if ((leftType == DataType::DATETIME || leftType == DataType::DATE) &&
                     (rightType == DataType::DATETIME || rightType == DataType::DATE))
            {
                ComparisonCondition tempCond("", ComparisonOperator::EQUAL, "");
                return tempCond.compareDateTime(leftValue, rightValue) != 0;
            }
            else
                return leftValue != rightValue;
        case ComparisonOperator::LESS_THAN:
            if (leftType == DataType::INT && rightType == DataType::INT)
                return std::stoi(leftValue) < std::stoi(rightValue);
            else if ((leftType == DataType::FLOAT || leftType == DataType::DOUBLE) &&
                     (rightType == DataType::FLOAT || rightType == DataType::DOUBLE))
                return std::stof(leftValue) < std::stof(rightValue);
            else if ((leftType == DataType::DATETIME || leftType == DataType::DATE) &&
                     (rightType == DataType::DATETIME || rightType == DataType::DATE))
            {
                ComparisonCondition tempCond("", ComparisonOperator::EQUAL, "");
                return tempCond.compareDateTime(leftValue, rightValue) < 0;
            }
            else
                return leftValue < rightValue;
        case ComparisonOperator::LESS_EQUAL:
            if (leftType == DataType::INT && rightType == DataType::INT)
                return std::stoi(leftValue) <= std::stoi(rightValue);
            else if ((leftType == DataType::FLOAT || leftType == DataType::DOUBLE) &&
                     (rightType == DataType::FLOAT || rightType == DataType::DOUBLE))
                return std::stof(leftValue) <= std::stof(rightValue);
            else if ((leftType == DataType::DATETIME || leftType == DataType::DATE) &&
                     (rightType == DataType::DATETIME || rightType == DataType::DATE))
            {
                ComparisonCondition tempCond("", ComparisonOperator::EQUAL, "");
                return tempCond.compareDateTime(leftValue, rightValue) <= 0;
            }
            else
                return leftValue <= rightValue;
        case ComparisonOperator::GREATER_THAN:
            if (leftType == DataType::INT && rightType == DataType::INT)
                return std::stoi(leftValue) > std::stoi(rightValue);
            else if ((leftType == DataType::FLOAT || leftType == DataType::DOUBLE) &&
                     (rightType == DataType::FLOAT || rightType == DataType::DOUBLE))
                return std::stof(leftValue) > std::stof(rightValue);
            else if ((leftType == DataType::DATETIME || leftType == DataType::DATE) &&
                     (rightType == DataType::DATETIME || rightType == DataType::DATE))
            {
                ComparisonCondition tempCond("", ComparisonOperator::EQUAL, "");
                return tempCond.compareDateTime(leftValue, rightValue) > 0;
            }
            else
                return leftValue > rightValue;
        case ComparisonOperator::GREATER_EQUAL:
            if (leftType == DataType::INT && rightType == DataType::INT)
                return std::stoi(leftValue) >= std::stoi(rightValue);
            else if ((leftType == DataType::FLOAT || leftType == DataType::DOUBLE) &&
                     (rightType == DataType::FLOAT || rightType == DataType::DOUBLE))
                return std::stof(leftValue) >= std::stof(rightValue);
            else if ((leftType == DataType::DATETIME || leftType == DataType::DATE) &&
                     (rightType == DataType::DATETIME || rightType == DataType::DATE))
            {
                ComparisonCondition tempCond("", ComparisonOperator::EQUAL, "");
                return tempCond.compareDateTime(leftValue, rightValue) >= 0;
            }
            else
                return leftValue >= rightValue;
        default:
            return false;
        }
    }

    std::string ColumnComparisonCondition::toString() const
    {
        std::string opStr;
        switch (m_operator)
        {
        case ComparisonOperator::EQUAL:
            opStr = " = ";
            break;
        case ComparisonOperator::NOT_EQUAL:
            opStr = " != ";
            break;
        case ComparisonOperator::LESS_THAN:
            opStr = " < ";
            break;
        case ComparisonOperator::LESS_EQUAL:
            opStr = " <= ";
            break;
        case ComparisonOperator::GREATER_THAN:
            opStr = " > ";
            break;
        case ComparisonOperator::GREATER_EQUAL:
            opStr = " >= ";
            break;
        default:
            opStr = " ?? ";
            break;
        }

        return m_leftColumn + opStr + m_rightColumn;
    }

    std::string ColumnComparisonCondition::getCUDACondition() const
    {
        std::string opStr;
        switch (m_operator)
        {
        case ComparisonOperator::EQUAL:
            opStr = "==";
            break;
        case ComparisonOperator::NOT_EQUAL:
            opStr = "!=";
            break;
        case ComparisonOperator::LESS_THAN:
            opStr = "<";
            break;
        case ComparisonOperator::LESS_EQUAL:
            opStr = "<=";
            break;
        case ComparisonOperator::GREATER_THAN:
            opStr = ">";
            break;
        case ComparisonOperator::GREATER_EQUAL:
            opStr = ">=";
            break;
        default:
            opStr = "??";
            break;
        }

        return "(" + m_leftColumn + " " + opStr + " " + m_rightColumn + ")";
    }

    std::unique_ptr<Condition> ColumnComparisonCondition::clone() const
    {
        return std::make_unique<ColumnComparisonCondition>(m_leftColumn, m_operator, m_rightColumn);
    }

    // Implement the new ConditionBuilder methods
    std::unique_ptr<Condition> ConditionBuilder::columnEquals(const std::string &leftColumn, const std::string &rightColumn)
    {
        return std::make_unique<ColumnComparisonCondition>(leftColumn, ComparisonOperator::EQUAL, rightColumn);
    }

    std::unique_ptr<Condition> ConditionBuilder::columnNotEquals(const std::string &leftColumn, const std::string &rightColumn)
    {
        return std::make_unique<ColumnComparisonCondition>(leftColumn, ComparisonOperator::NOT_EQUAL, rightColumn);
    }

    std::unique_ptr<Condition> ConditionBuilder::columnLessThan(const std::string &leftColumn, const std::string &rightColumn)
    {
        return std::make_unique<ColumnComparisonCondition>(leftColumn, ComparisonOperator::LESS_THAN, rightColumn);
    }

    std::unique_ptr<Condition> ConditionBuilder::columnLessEqual(const std::string &leftColumn, const std::string &rightColumn)
    {
        return std::make_unique<ColumnComparisonCondition>(leftColumn, ComparisonOperator::LESS_EQUAL, rightColumn);
    }

    std::unique_ptr<Condition> ConditionBuilder::columnGreaterThan(const std::string &leftColumn, const std::string &rightColumn)
    {
        return std::make_unique<ColumnComparisonCondition>(leftColumn, ComparisonOperator::GREATER_THAN, rightColumn);
    }

    std::unique_ptr<Condition> ConditionBuilder::columnGreaterEqual(const std::string &leftColumn, const std::string &rightColumn)
    {
        return std::make_unique<ColumnComparisonCondition>(leftColumn, ComparisonOperator::GREATER_EQUAL, rightColumn);
    }
} // namespace GPUDBMS