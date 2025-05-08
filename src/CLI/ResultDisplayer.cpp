#include "../../include/CLI/ResultDisplayer.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <string>

namespace GPUDBMS
{

    void ResultDisplayer::displayTable(const Table &table)
    {
        const size_t columnCount = table.getColumnCount();
        const size_t rowCount = table.getRowCount();

        if (columnCount == 0)
        {
            std::cout << "Empty table (no columns)" << std::endl;
            return;
        }

        if (rowCount == 0)
        {
            std::cout << "Empty result set (0 rows)" << std::endl;
            displayHeader(table);
            return;
        }

        // Calculate column widths based on data and headers
        std::vector<size_t> columnWidths = calculateColumnWidths(table);

        // Display the header
        displayHeaderWithWidths(table, columnWidths);

        // Display separator line
        displaySeparator(columnWidths);

        // Display data rows
        displayRows(table, columnWidths);
    }

    std::vector<size_t> ResultDisplayer::calculateColumnWidths(const Table &table)
    {
        const size_t columnCount = table.getColumnCount();
        const size_t rowCount = std::min(table.getRowCount(), m_maxRowsToConsider);

        std::vector<size_t> columnWidths(columnCount);

        // Initialize with header lengths
        for (size_t col = 0; col < columnCount; ++col)
        {
            columnWidths[col] = table.getColumnName(col).length();
        }

        // Consider data in each column
        for (size_t row = 0; row < rowCount; ++row)
        {
            for (size_t col = 0; col < columnCount; ++col)
            {
                std::string cellValue = getCellValueAsString(table, col, row);
                columnWidths[col] = std::max(columnWidths[col], cellValue.length());
            }
        }

        // Apply minimum and maximum column width constraints
        for (size_t col = 0; col < columnCount; ++col)
        {
            columnWidths[col] = std::max(columnWidths[col], m_minColumnWidth);
            columnWidths[col] = std::min(columnWidths[col], m_maxColumnWidth);
        }

        return columnWidths;
    }

    void ResultDisplayer::displayHeader(const Table &table)
    {
        const size_t columnCount = table.getColumnCount();

        for (size_t col = 0; col < columnCount; ++col)
        {
            if (col > 0)
                std::cout << " | ";
            std::cout << table.getColumnName(col);
        }
        std::cout << std::endl;

        // Display separator
        for (size_t col = 0; col < columnCount; ++col)
        {
            if (col > 0)
                std::cout << "-+-";
            std::cout << std::string(table.getColumnName(col).length(), '-');
        }
        std::cout << std::endl;
    }

    void ResultDisplayer::displayHeaderWithWidths(const Table &table, const std::vector<size_t> &columnWidths)
    {
        const size_t columnCount = table.getColumnCount();
    
        // Display top border
        std::cout << "+";
        for (size_t col = 0; col < columnCount; ++col)
        {
            if (col > 0)
                std::cout << "+";
            std::cout << std::string(columnWidths[col] + 2, '-');
        }
        std::cout << "+" << std::endl;
    
        // Display column headers
        std::cout << "| ";
        for (size_t col = 0; col < columnCount; ++col)
        {
            if (col > 0)
                std::cout << " | ";
            std::cout << std::left << std::setw(columnWidths[col]) << table.getColumnName(col);
        }
        std::cout << " |" << std::endl;
    
        // Display header-data separator
        std::cout << "+";
        for (size_t col = 0; col < columnCount; ++col)
        {
            if (col > 0)
                std::cout << "+";
            std::cout << std::string(columnWidths[col] + 2, '-');
        }
        std::cout << "+" << std::endl;
    }
    
    void ResultDisplayer::displaySeparator(const std::vector<size_t>& columnWidths) {
        std::cout << "+";
        for (size_t col = 0; col < columnWidths.size(); ++col) {
            if (col > 0) std::cout << "+";
            std::cout << std::string(columnWidths[col] + 2, '-');
        }
        std::cout << "+" << std::endl;
    }

    void ResultDisplayer::displayRows(const Table &table, const std::vector<size_t> &columnWidths)
    {
        const size_t columnCount = table.getColumnCount();
        const size_t rowCount = table.getRowCount();
        const size_t displayRows = std::min(rowCount, m_maxDisplayRows);
        
        // Display data rows
        for (size_t row = 0; row < displayRows; ++row)
        {
            // Use consistent format for all rows
            std::cout << "| ";
            for (size_t col = 0; col < columnCount; ++col)
            {
                if (col > 0)
                    std::cout << " | ";
                
                std::string cellValue = getCellValueAsString(table, col, row);
                if (cellValue.length() > m_maxColumnWidth)
                {
                    cellValue = cellValue.substr(0, m_maxColumnWidth - 3) + "...";
                }
                
                std::cout << std::left << std::setw(columnWidths[col]) << cellValue;
            }
            std::cout << " |" << std::endl;
        }
    
        // If we truncated the output, indicate that
        if (displayRows < rowCount)
        {
            // Add a separator before the "..." row if separators are enabled
            if (m_showRowSeparators && displayRows > 0)
            {
                displaySeparator(columnWidths);
            }
    
            std::cout << "| ";
            for (size_t col = 0; col < columnCount; ++col)
            {
                if (col > 0)
                    std::cout << " | ";
                std::cout << std::left << std::setw(columnWidths[col]) << "...";
            }
            std::cout << " |" << std::endl;
        }
    
        // Display bottom border
        std::cout << "+";
        for (size_t col = 0; col < columnCount; ++col)
        {
            if (col > 0)
                std::cout << "+";
            std::cout << std::string(columnWidths[col] + 2, '-');
        }
        std::cout << "+" << std::endl;
    }

    std::string ResultDisplayer::getCellValueAsString(const Table &table, size_t col, size_t row)
    {
        const Column &column = table.getColumns()[col];

        try
        {
            switch (column.getType())
            {
            case DataType::INT:
                return std::to_string(table.getIntValue(col, row));

            case DataType::FLOAT:
                return formatFloat(table.getFloatValue(col, row));

            case DataType::DOUBLE:
                return formatDouble(table.getDoubleValue(col, row));

            case DataType::VARCHAR:
            case DataType::STRING:
                return table.getStringValue(col, row);

            case DataType::BOOL:
                return table.getBoolValue(col, row) ? "true" : "false";

            case DataType::DATE:
            case DataType::DATETIME:
                return table.getStringValue(col, row);

            default:
                return "N/A";
            }
        }
        catch (const std::exception &e)
        {
            return "ERROR";
        }
    }

    std::string ResultDisplayer::formatFloat(float value)
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(m_floatPrecision) << value;
        std::string s = ss.str();

        // Remove trailing zeros
        s.erase(s.find_last_not_of('0') + 1, std::string::npos);
        if (s.back() == '.')
        {
            s.pop_back();
        }

        return s;
    }

    std::string ResultDisplayer::formatDouble(double value)
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(m_doublePrecision) << value;
        std::string s = ss.str();

        // Remove trailing zeros
        s.erase(s.find_last_not_of('0') + 1, std::string::npos);
        if (s.back() == '.')
        {
            s.pop_back();
        }

        return s;
    }

} // namespace GPUDBMS