#include "QueryProcessing/SubqueryHandler.hpp"
#include "Utilities/Logger.hpp"
#include "Utilities/ErrorHandling.hpp"
#include "Utilities/StringUtils.hpp"

namespace SQLQueryProcessor {

SubqueryHandler::SubqueryHandler(QueryExecutor& queryExecutor)
    : queryExecutor(queryExecutor) {
}

std::shared_ptr<Table> SubqueryHandler::processSubquery(
    const std::string& fullQuery, 
    size_t& subqueryStart, 
    size_t& subqueryEnd) {
    
    Logger::debug("Processing subquery");
    
    if (!hasSubquery(fullQuery, subqueryStart, subqueryEnd)) {
        return nullptr;
    }
    
    // Extract the subquery
    std::string subquery = extractSubquery(fullQuery, subqueryStart, subqueryEnd);
    
    // Execute the subquery
    QueryExecutor::ExecutionStats subStats;
    std::shared_ptr<Table> result = queryExecutor.executeQuery(subquery, subStats);
    
    if (!result) {
        throw ExecutionException("Subquery execution failed");
    }
    
    Logger::debug("Subquery processed successfully, returned " + 
                 std::to_string(result->getRowCount()) + " rows");
    
    return result;
}

bool SubqueryHandler::hasSubquery(const std::string& query, size_t& start, size_t& end) {
    // Find opening parenthesis that might start a subquery
    start = query.find('(');
    
    while (start != std::string::npos) {
        // Look for "SELECT" keyword inside parentheses
        size_t selectPos = query.find("SELECT", start);
        size_t selectPosUpper = query.find("select", start);
        
        if (selectPos == std::string::npos && selectPosUpper == std::string::npos) {
            return false;
        }
        
        // Use the first occurrence of SELECT/select
        size_t selectKeyword = (selectPos == std::string::npos) ? selectPosUpper :
                              ((selectPosUpper == std::string::npos) ? selectPos :
                               std::min(selectPos, selectPosUpper));
        
        // Find matching closing parenthesis
        end = findMatchingParenthesis(query, start);
        
        if (end == std::string::npos) {
            throw ParsingException("Mismatched parentheses in query");
        }
        
        // Check if SELECT is inside these parentheses
        if (selectKeyword > start && selectKeyword < end) {
            return true;
        }
        
        // Try next opening parenthesis
        start = query.find('(', start + 1);
    }
    
    return false;
}

std::string SubqueryHandler::replaceSubqueryWithTemp(
    const std::string& query, 
    const std::string& tempTableName, 
    size_t subqueryStart, 
    size_t subqueryEnd) {
    
    // Replace the subquery with the temporary table name
    return query.substr(0, subqueryStart) + tempTableName + query.substr(subqueryEnd + 1);
}

std::string SubqueryHandler::extractSubquery(
    const std::string& query, 
    size_t start, 
    size_t end) {
    
    // Extract the subquery without the outermost parentheses
    return query.substr(start + 1, end - start - 1);
}

int SubqueryHandler::findMatchingParenthesis(const std::string& query, size_t openPos) {
    int depth = 1;
    
    for (size_t i = openPos + 1; i < query.length(); ++i) {
        if (query[i] == '(') {
            depth++;
        } else if (query[i] == ')') {
            depth--;
            if (depth == 0) {
                return i;
            }
        }
    }
    
    return -1;  // No matching closing parenthesis found
}

} // namespace SQLQueryProcessor