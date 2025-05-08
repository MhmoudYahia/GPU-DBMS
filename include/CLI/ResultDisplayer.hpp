#pragma once

#include <string>
#include <vector>
#include <sstream>
#include "../DataHandling/Table.hpp"

namespace GPUDBMS {

/**
 * @class ResultDisplayer
 * @brief Formats and displays query results in a human-readable format
 */
class ResultDisplayer {
public:
    /**
     * @brief Display a table in formatted output
     * @param table The table to display
     */
    void displayTable(const Table& table);

private:
    // Display configuration
    const size_t m_minColumnWidth = 3;
    const size_t m_maxColumnWidth = 30;
    const size_t m_maxRowsToConsider = 100;  // For width calculation
    const size_t m_maxDisplayRows = 100;     // For output
    const int m_floatPrecision = 4;
    const int m_doublePrecision = 6;
    bool m_showRowSeparators = false; // Set to true if you want separators between each row
    
    // Helper methods
    std::vector<size_t> calculateColumnWidths(const Table& table);
    void displayHeader(const Table& table);
    void displayHeaderWithWidths(const Table& table, const std::vector<size_t>& columnWidths);
    void displaySeparator(const std::vector<size_t>& columnWidths);
    void displayRows(const Table& table, const std::vector<size_t>& columnWidths);
    std::string getCellValueAsString(const Table& table, size_t col, size_t row);
    std::string formatFloat(float value);
    std::string formatDouble(double value);
};

} // namespace GPUDBMS