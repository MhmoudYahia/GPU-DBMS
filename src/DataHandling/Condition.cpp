#include "../../include/DataHandling/Condition.hpp"
#include <algorithm>
#include <regex>
#include <sstream>

namespace GPUDBMS
{

    // ComparisonCondition implementation
    ComparisonCondition::ComparisonCondition(const std::string &columnName, ComparisonOperator op, const std::string &value)
        : m_columnName(columnName), m_operator(op), m_value(value)
    {
    }

    bool ComparisonCondition::evaluate(const std::vector<std::string> &row, std::unordered_map<std::string, int> columnNameToIndex) const
    {
        // Find the index of the column in columnIndices
        auto it = columnNameToIndex.find(m_columnName);

        if (it == columnNameToIndex.end())
        {
            return false; // Column not found
        }

        int index = it->second;
        if (index >= row.size())
        {
            return false; // Invalid index
        }

        const std::string &cellValue = row[index];

        switch (m_operator)
        {
        case ComparisonOperator::EQUAL:
            return cellValue == m_value;
        case ComparisonOperator::NOT_EQUAL:
            return cellValue != m_value;
        case ComparisonOperator::LESS_THAN:
            return cellValue < m_value;
        case ComparisonOperator::LESS_EQUAL:
            return cellValue <= m_value;
        case ComparisonOperator::GREATER_THAN:
            return cellValue > m_value;
        case ComparisonOperator::GREATER_EQUAL:
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

    bool LogicalCondition::evaluate(const std::vector<std::string> &row, std::unordered_map<std::string, int> columnNameToIndex) const
    {
        switch (m_operator)
        {
        case LogicalOperator::AND:
            return m_left->evaluate(row, columnNameToIndex) && m_right->evaluate(row, columnNameToIndex);
        case LogicalOperator::OR:
            return m_left->evaluate(row, columnNameToIndex) || m_right->evaluate(row, columnNameToIndex);
        case LogicalOperator::NOT:
            return !m_left->evaluate(row, columnNameToIndex);
        default:
            return false;
        }
    }

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

} // namespace GPUDBMS