#pragma once
#include "DataHandling/Table.hpp"
#include "DataHandling/StorageManager.hpp"
#include "QueryProcessing/QueryExecutor.hpp"
#include <string>
#include <memory>
#include <vector>
#include <iostream>

namespace SQLQueryProcessor
{

    class ResultDisplayer
    {
    public:
        ResultDisplayer(StorageManager &storageManager);
        ~ResultDisplayer() = default;

        // Display a result table in the console
        void displayTable(const std::shared_ptr<Table> &table, std::ostream &out = std::cout) const;

        // Display execution statistics
        void displayStats(const QueryExecutor::ExecutionStats &stats, std::ostream &out = std::cout) const;

        // Save results to files
        void saveResults(const std::shared_ptr<Table> &table, const std::string &baseFileName) const;

        // Display an error message
        void displayError(const std::string &errorMessage, std::ostream &out = std::cerr) const;

        // Display a success message
        void displaySuccess(const std::string &message, std::ostream &out = std::cout) const;

        // Set maximum displayed rows (for large results)
        void setMaxDisplayedRows(size_t maxRows);
        size_t getMaxDisplayedRows() const;

    private:
        StorageManager &storageManager;
        size_t maxDisplayedRows;

        // Helper methods for formatting
        std::vector<size_t> calculateColumnWidths(const std::shared_ptr<Table> &table) const;
        std::string truncateValue(const std::string &value, size_t maxWidth) const;
        std::string formatDuration(double milliseconds) const;
        std::string formatTimestamp() const;
    };

} // namespace SQLQueryProcessor