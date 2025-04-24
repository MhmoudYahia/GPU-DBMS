#include "CLI/ResultDisplayer.hpp"
#include "Utilities/Logger.hpp"
#include "Utilities/StringUtils.hpp"
#include <iomanip>
#include <chrono>
#include <ctime>

namespace SQLQueryProcessor {

ResultDisplayer::ResultDisplayer(StorageManager& storageManager)
    : storageManager(storageManager), maxDisplayedRows(100) {
}

void ResultDisplayer::displayTable(const std::shared_ptr<Table>& table, std::ostream& out) const {
    if (!table) {
        out << "No results to display" << std::endl;
        return;
    }
    
    const auto& columnNames = table->getColumnNames();
    size_t numColumns = columnNames.size();
    size_t numRows = table->getRowCount();
    
    // Calculate column widths
    std::vector<size_t> columnWidths = calculateColumnWidths(table);
    
    // Calculate total width
    size_t totalWidth = 1;  // Starting border
    for (size_t width : columnWidths) {
        totalWidth += width + 3;  // Column width + separator
    }
    
    // Print top border
    out << std::string(totalWidth, '-') << std::endl;
    
    // Print header
    out << "|";
    for (size_t i = 0; i < numColumns; ++i) {
        out << " " << std::setw(columnWidths[i]) << std::left << truncateValue(columnNames[i], columnWidths[i]) << " |";
    }
    out << std::endl;
    
    // Print header separator
    out << "|";
    for (size_t i = 0; i < numColumns; ++i) {
        out << std::string(columnWidths[i] + 2, '-') << "|";
    }
    out << std::endl;
    
    // Print data rows
    size_t rowsToDisplay = std::min(numRows, maxDisplayedRows);
    for (size_t i = 0; i < rowsToDisplay; ++i) {
        out << "|";
        for (size_t j = 0; j < numColumns; ++j) {
            out << " " << std::setw(columnWidths[j]) << std::left 
                << truncateValue(table->getValue(i, j), columnWidths[j]) << " |";
        }
        out << std::endl;
    }
    
    // Print bottom border
    out << std::string(totalWidth, '-') << std::endl;
    
    // Print row count info
    out << rowsToDisplay << " of " << numRows << " rows displayed" << std::endl;
}

void ResultDisplayer::displayStats(const QueryExecutor::ExecutionStats& stats, std::ostream& out) const {
    out << "Execution Statistics:" << std::endl;
    out << "  Parsing Time:   " << formatDuration(stats.parsingTimeMs) << std::endl;
    out << "  Execution Time: " << formatDuration(stats.executionTimeMs) << std::endl;
    out << "  Output Time:    " << formatDuration(stats.outputTimeMs) << std::endl;
    out << "  Total Time:     " << formatDuration(stats.totalTimeMs) << std::endl;
    out << "  Result Size:    " << stats.resultSize << " rows" << std::endl;
}

void ResultDisplayer::saveResults(const std::shared_ptr<Table>& table, const std::string& baseFileName) const {
    if (!table) {
        Logger::warning("Attempted to save null result table");
        return;
    }
    
    // Save as CSV
    storageManager.saveResultCSV(table, baseFileName);
    
    // Save as TXT (formatted table)
    std::ostringstream oss;
    oss << "Query Result: " << baseFileName << std::endl;
    oss << "Generated at: " << formatTimestamp() << std::endl << std::endl;
    
    displayTable(table, oss);
    
    storageManager.saveResultTXT(oss.str(), baseFileName);
    
    Logger::info("Saved result to " + baseFileName + ".csv and " + baseFileName + ".txt");
}

void ResultDisplayer::displayError(const std::string& errorMessage, std::ostream& out) const {
    out << "ERROR: " << errorMessage << std::endl;
}

void ResultDisplayer::displaySuccess(const std::string& message, std::ostream& out) const {
    out << "SUCCESS: " << message << std::endl;
}

void ResultDisplayer::setMaxDisplayedRows(size_t maxRows) {
    maxDisplayedRows = maxRows;
}

size_t ResultDisplayer::getMaxDisplayedRows() const {
    return maxDisplayedRows;
}

std::vector<size_t> ResultDisplayer::calculateColumnWidths(const std::shared_ptr<Table>& table) const {
    const size_t MIN_WIDTH = 5;
    const size_t MAX_WIDTH = 30;
    
    const auto& columnNames = table->getColumnNames();
    size_t numColumns = columnNames.size();
    size_t numRows = table->getRowCount();
    
    std::vector<size_t> widths(numColumns, MIN_WIDTH);
    
    // Check header widths
    for (size_t i = 0; i < numColumns; ++i) {
        widths[i] = std::max(widths[i], std::min(columnNames[i].length(), MAX_WIDTH));
    }
    
    // Check data widths (using a sample of rows)
    size_t sampleSize = std::min(numRows, static_cast<size_t>(100));
    for (size_t i = 0; i < sampleSize; ++i) {
        for (size_t j = 0; j < numColumns; ++j) {
            widths[j] = std::max(widths[j], std::min(table->getValue(i, j).length(), MAX_WIDTH));
        }
    }
    
    return widths;
}

std::string ResultDisplayer::truncateValue(const std::string& value, size_t maxWidth) const {
    if (value.length() <= maxWidth) {
        return value;
    }
    
    if (maxWidth <= 3) {
        return value.substr(0, maxWidth);
    }
    
    return value.substr(0, maxWidth - 3) + "...";
}

std::string ResultDisplayer::formatDuration(double milliseconds) const {
    if (milliseconds < 1.0) {
        return StringUtils::formatDouble(milliseconds * 1000, 2) + " μs";
    } else if (milliseconds < 1000.0) {
        return StringUtils::formatDouble(milliseconds, 2) + " ms";
    } else {
        return StringUtils::formatDouble(milliseconds / 1000.0, 2) + " s";
    }
}

std::string ResultDisplayer::formatTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::tm tm_buf;
    #ifdef _WIN32
        localtime_s(&tm_buf, &now_time_t);
    #else
        localtime_r(&now_time_t, &tm_buf);
    #endif
    
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
    
    return oss.str();
}

} // namespace SQLQueryProcessor