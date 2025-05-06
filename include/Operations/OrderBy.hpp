#ifndef ORDERBY_HPP
#define ORDERBY_HPP

#include <vector>
#include <string>
#include "../DataHandling/Table.hpp"
#include "../Operations/OrderByGPU.cuh"

namespace GPUDBMS
{
   

    /**
     * @class OrderBy
     * @brief Orders the rows of a table based on specified columns
     */
    class OrderBy
    {
    public:
        /**
         * @brief Construct a new OrderBy operation
         * 
         * @param inputTable The input table to sort
         * @param sortColumn The column name to sort by
         * @param order The sort order (ASC or DESC)
         */
        OrderBy(const Table &inputTable, 
                const std::string &sortColumn,
                SortOrder order = SortOrder::ASC);

        /**
         * @brief Construct a new OrderBy operation
         * 
         * @param inputTable The input table to sort
         * @param sortColumns Vector of column names to sort by
         * @param sortOrders Vector of sort orders for each column (ASC or DESC)
         */
        OrderBy(const Table &inputTable, 
                const std::vector<std::string> &sortColumns,
                const std::vector<SortOrder> &sortOrders);

        /**
         * @brief Execute the order by operation using GPU acceleration
         * 
         * @return Table The sorted result table
         */
        Table execute(bool useGPU = false);

        /**
         * @brief Execute the order by operation on CPU (fallback)
         * 
         * @return Table The sorted result table
         */
        Table executeCPU();

        Table executeGPU();

    private:
        const Table &m_inputTable;
        std::vector<std::string> m_sortColumns;
        std::vector<SortOrder> m_sortOrders;
        std::vector<int> m_columnIndices;
        
        /**
         * @brief Resolve column names to indices
         */
        void resolveColumnIndices();
        
        /**
         * @brief Compare two rows based on sort columns and orders
         * 
         * @param rowIndexA Index of the first row
         * @param rowIndexB Index of the second row
         * @return true if rowA should come before rowB
         */
        bool compareRows(size_t rowIndexA, size_t rowIndexB) const;
    };

} // namespace GPUDBMS

#endif // ORDERBY_HPP