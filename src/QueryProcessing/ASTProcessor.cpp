#include "QueryProcessing/ASTProcessor.hpp"
#include "Utilities/Logger.hpp"
#include "Utilities/ErrorHandling.hpp"
#include "Utilities/StringUtils.hpp"
#include <regex>
#include <sstream>
#include <algorithm>

namespace SQLQueryProcessor {

ASTProcessor::ASTProcessor(StorageManager& storageManager)
    : storageManager(storageManager) {
}

ASTProcessor::QueryInfo ASTProcessor::processQuery(const std::string& query) {
    Logger::debug("Processing query: " + query);
    
    QueryInfo queryInfo;
    
    // Check if query has subquery
    queryInfo.subQuery = extractSubquery(query);
    queryInfo.hasSubquery = !queryInfo.subQuery.empty();
    
    // Extract and parse different clauses
    std::string processedQuery = queryInfo.hasSubquery ? 
                                query.substr(0, query.find("(")) + 
                                "SUBQUERY_RESULT" + 
                                query.substr(query.rfind(")") + 1) : 
                                query;
    
    std::string selectClause = extractClause(processedQuery, "SELECT", "FROM");
    std::string fromClause = extractClause(processedQuery, "FROM", "WHERE|GROUP BY|ORDER BY|$");
    std::string whereClause = extractClause(processedQuery, "WHERE", "GROUP BY|ORDER BY|$");
    std::string groupByClause = extractClause(processedQuery, "GROUP BY", "ORDER BY|$");
    std::string orderByClause = extractClause(processedQuery, "ORDER BY", "$");
    
    // Parse individual clauses
    queryInfo.selectColumns = parseSelectClause(selectClause);
    queryInfo.fromTables = parseFromClause(fromClause);
    
    // Extract join conditions and filter conditions from WHERE clause
    queryInfo.joins = parseJoinConditions(whereClause, queryInfo.fromTables);
    parseWhereConditions(whereClause, queryInfo);
    
    queryInfo.groupByColumns = parseGroupByClause(groupByClause);
    queryInfo.orderByColumns = parseOrderByClause(orderByClause);
    
    return queryInfo;
}

std::string ASTProcessor::extractSubquery(const std::string& query) {
    // Find the first occurrence of "("
    size_t start = query.find('(');
    if (start == std::string::npos) {
        return "";
    }
    
    // Count opening and closing parentheses to find matching closing parenthesis
    int count = 1;
    size_t end = start + 1;
    
    while (end < query.length() && count > 0) {
        if (query[end] == '(') {
            count++;
        } else if (query[end] == ')') {
            count--;
        }
        end++;
    }
    
    if (count != 0) {
        throw ParsingException("Mismatched parentheses in query");
    }
    
    // Extract the subquery (without the outer parentheses)
    std::string subquery = query.substr(start + 1, end - start - 2);
    
    // Verify it's actually a subquery by checking for SELECT
    if (StringUtils::toUpper(subquery).find("SELECT") == std::string::npos) {
        return "";  // Not a subquery
    }
    
    return subquery;
}

std::vector<ASTProcessor::ColumnInfo> ASTProcessor::parseSelectClause(const std::string& selectClause) {
    std::vector<ColumnInfo> columns;
    
    // Split by commas
    std::vector<std::string> columnExpressions = tokenizeClause(selectClause, ",");
    
    for (const auto& expr : columnExpressions) {
        std::string trimmedExpr = StringUtils::trim(expr);
        ColumnInfo columnInfo;
        
        // Check for aggregate functions
        std::regex aggrRegex("(COUNT|SUM|AVG|MIN|MAX)\\s*\\((.+?)\\)\\s*(?:AS\\s+(.+?))?$",
                            std::regex_constants::icase);
        std::smatch matches;
        
        if (std::regex_search(trimmedExpr, matches, aggrRegex)) {
            columnInfo.isAggregation = true;
            
            // Determine aggregate function
            std::string funcName = StringUtils::toUpper(matches[1].str());
            if (funcName == "COUNT") columnInfo.aggrFunc = AggregateFunction::COUNT;
            else if (funcName == "SUM") columnInfo.aggrFunc = AggregateFunction::SUM;
            else if (funcName == "AVG") columnInfo.aggrFunc = AggregateFunction::AVG;
            else if (funcName == "MIN") columnInfo.aggrFunc = AggregateFunction::MIN;
            else if (funcName == "MAX") columnInfo.aggrFunc = AggregateFunction::MAX;
            
            // Extract column name
            std::string colExpr = StringUtils::trim(matches[2].str());
            if (colExpr == "*") {
                columnInfo.column = "*";
            } else {
                // Check if column has table prefix
                size_t dotPos = colExpr.find('.');
                if (dotPos != std::string::npos) {
                    columnInfo.table = StringUtils::trim(colExpr.substr(0, dotPos));
                    columnInfo.column = StringUtils::trim(colExpr.substr(dotPos + 1));
                } else {
                    columnInfo.column = colExpr;
                }
            }
            
            // Extract alias if present
            if (matches.size() > 3 && matches[3].matched) {
                columnInfo.alias = StringUtils::trim(matches[3].str());
            } else {
                // Generate default alias
                columnInfo.alias = funcName + "(" + columnInfo.column + ")";
            }
        }
        // Check for normal columns with possible table prefix and alias
        else {
            // Check for alias
            size_t asPos = StringUtils::toUpper(trimmedExpr).find(" AS ");
            if (asPos != std::string::npos) {
                std::string colExpr = StringUtils::trim(trimmedExpr.substr(0, asPos));
                columnInfo.alias = StringUtils::trim(trimmedExpr.substr(asPos + 4));
                
                // Check if column has table prefix
                size_t dotPos = colExpr.find('.');
                if (dotPos != std::string::npos) {
                    columnInfo.table = StringUtils::trim(colExpr.substr(0, dotPos));
                    columnInfo.column = StringUtils::trim(colExpr.substr(dotPos + 1));
                } else {
                    columnInfo.column = colExpr;
                }
            } else {
                // No alias, check for table prefix
                size_t dotPos = trimmedExpr.find('.');
                if (dotPos != std::string::npos) {
                    columnInfo.table = StringUtils::trim(trimmedExpr.substr(0, dotPos));
                    columnInfo.column = StringUtils::trim(trimmedExpr.substr(dotPos + 1));
                    
                    // Use column name as alias
                    columnInfo.alias = columnInfo.column;
                } else {
                    columnInfo.column = trimmedExpr;
                    columnInfo.alias = trimmedExpr;
                }
            }
        }
        
        columns.push_back(columnInfo);
    }
    
    return columns;
}

std::vector<std::string> ASTProcessor::parseFromClause(const std::string& fromClause) {
    std::vector<std::string> tables;
    
    // Split by commas
    std::vector<std::string> tableExpressions = tokenizeClause(fromClause, ",");
    
    for (const auto& expr : tableExpressions) {
        std::string trimmedExpr = StringUtils::trim(expr);
        
        // Check for alias
        std::vector<std::string> parts = StringUtils::splitAndTrim(trimmedExpr, ' ');
        if (!parts.empty()) {
            tables.push_back(parts[0]);  // Use the table name only
        }
    }
    
    // Check for subquery placeholder
    auto it = std::find(tables.begin(), tables.end(), "SUBQUERY_RESULT");
    if (it != tables.end()) {
        *it = "SUBQUERY_RESULT";
    }
    
    return tables;
}

std::vector<ASTProcessor::JoinInfo> ASTProcessor::parseJoinConditions(
    const std::string& whereClause, 
    const std::vector<std::string>& tables) {
    
    std::vector<JoinInfo> joins;
    
    if (whereClause.empty() || tables.size() <= 1) {
        return joins;
    }
    
    // Split conditions by AND
    std::vector<std::string> conditions = tokenizeClause(whereClause, "AND");
    
    for (const auto& condition : conditions) {
        std::string trimmedCond = StringUtils::trim(condition);
        
        // Look for equality conditions that involve columns from different tables
        size_t eqPos = trimmedCond.find('=');
        if (eqPos == std::string::npos) {
            continue;
        }
        
        std::string leftExpr = StringUtils::trim(trimmedCond.substr(0, eqPos));
        std::string rightExpr = StringUtils::trim(trimmedCond.substr(eqPos + 1));
        
        // Check if both sides have table prefixes
        size_t leftDotPos = leftExpr.find('.');
        size_t rightDotPos = rightExpr.find('.');
        
        if (leftDotPos == std::string::npos || rightDotPos == std::string::npos) {
            continue;
        }
        
        std::string leftTable = StringUtils::trim(leftExpr.substr(0, leftDotPos));
        std::string leftColumn = StringUtils::trim(leftExpr.substr(leftDotPos + 1));
        std::string rightTable = StringUtils::trim(rightExpr.substr(0, rightDotPos));
        std::string rightColumn = StringUtils::trim(rightExpr.substr(rightDotPos + 1));
        
        // Check if the tables exist in the FROM clause
        if (std::find(tables.begin(), tables.end(), leftTable) != tables.end() &&
            std::find(tables.begin(), tables.end(), rightTable) != tables.end()) {
            
            JoinInfo join;
            join.leftTable = leftTable;
            join.rightTable = rightTable;
            join.leftColumn = leftColumn;
            join.rightColumn = rightColumn;
            
            joins.push_back(join);
        }
    }
    
    return joins;
}

void ASTProcessor::parseWhereConditions(const std::string& whereClause, QueryInfo& queryInfo) {
    if (whereClause.empty()) {
        return;
    }
    
    // Split by logical operators (AND, OR), preserving them
    std::vector<std::string> tokens;
    std::string remaining = whereClause;
    
    size_t pos;
    while (!remaining.empty()) {
        // Try to find the next logical operator
        size_t andPos = StringUtils::toUpper(remaining).find(" AND ");
        size_t orPos = StringUtils::toUpper(remaining).find(" OR ");
        
        if (andPos == std::string::npos && orPos == std::string::npos) {
            // No more operators, add the remaining as a condition
            tokens.push_back(StringUtils::trim(remaining));
            break;
        }
        
        // Determine which operator comes first
        if (andPos != std::string::npos && (orPos == std::string::npos || andPos < orPos)) {
            tokens.push_back(StringUtils::trim(remaining.substr(0, andPos)));
            tokens.push_back("AND");
            remaining = remaining.substr(andPos + 5);
        } else {
            tokens.push_back(StringUtils::trim(remaining.substr(0, orPos)));
            tokens.push_back("OR");
            remaining = remaining.substr(orPos + 4);
        }
    }
    
    // Process tokens to extract conditions and logical operators
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i % 2 == 0) {
            // This is a condition
            std::string condition = tokens[i];
            
            // Skip join conditions (handled separately)
            if (condition.find('.') != std::string::npos && 
                condition.find('=') != std::string::npos) {
                
                std::string leftExpr = StringUtils::trim(condition.substr(0, condition.find('=')));
                std::string rightExpr = StringUtils::trim(condition.substr(condition.find('=') + 1));
                
                if (leftExpr.find('.') != std::string::npos && 
                    rightExpr.find('.') != std::string::npos) {
                    continue;  // This is likely a join condition
                }
            }
            
            // Parse the condition
            ComparisonOperator op;
            size_t opPos = std::string::npos;
            
            // Find which operator is present
            if ((opPos = condition.find("=")) != std::string::npos) {
                op = ComparisonOperator::EQUAL;
            } else if ((opPos = condition.find("!=")) != std::string::npos ||
                       (opPos = condition.find("<>")) != std::string::npos) {
                op = ComparisonOperator::NOT_EQUAL;
            } else if ((opPos = condition.find(">=")) != std::string::npos) {
                op = ComparisonOperator::GREATER_EQUAL;
            } else if ((opPos = condition.find("<=")) != std::string::npos) {
                op = ComparisonOperator::LESS_EQUAL;
            } else if ((opPos = condition.find(">")) != std::string::npos) {
                op = ComparisonOperator::GREATER;
            } else if ((opPos = condition.find("<")) != std::string::npos) {
                op = ComparisonOperator::LESS;
            } else {
                continue;  // No valid operator found
            }
            
            std::string leftExpr = StringUtils::trim(condition.substr(0, opPos));
            std::string rightExpr = StringUtils::trim(condition.substr(opPos + (op == ComparisonOperator::NOT_EQUAL ? 2 : 1)));
            
            // Extract column name and value
            std::string columnName;
            size_t dotPos = leftExpr.find('.');
            if (dotPos != std::string::npos) {
                // Column has table prefix
                columnName = leftExpr.substr(dotPos + 1);
            } else {
                columnName = leftExpr;
            }
            
            // Remove quotes from value if present
            if ((rightExpr.front() == '\'' && rightExpr.back() == '\'') ||
                (rightExpr.front() == '"' && rightExpr.back() == '"')) {
                rightExpr = rightExpr.substr(1, rightExpr.length() - 2);
            }
            
            FilterCondition filterCond(columnName, op, rightExpr);
            queryInfo.whereConditions.push_back(filterCond);
            
            // Add logical operator if there is a next condition
            if (i + 1 < tokens.size()) {
                if (tokens[i + 1] == "AND") {
                    queryInfo.whereLogicalOps.push_back(LogicalOperator::AND);
                } else if (tokens[i + 1] == "OR") {
                    queryInfo.whereLogicalOps.push_back(LogicalOperator::OR);
                }
            }
        }
    }
}

std::vector<std::string> ASTProcessor::parseGroupByClause(const std::string& groupByClause) {
    std::vector<std::string> groupByColumns;
    
    if (groupByClause.empty()) {
        return groupByColumns;
    }
    
    // Split by commas
    std::vector<std::string> columnExpressions = tokenizeClause(groupByClause, ",");
    
    for (const auto& expr : columnExpressions) {
        std::string trimmedExpr = StringUtils::trim(expr);
        
        // Handle possible table prefix
        size_t dotPos = trimmedExpr.find('.');
        if (dotPos != std::string::npos) {
            // Extract column name only (ignore table prefix)
            groupByColumns.push_back(StringUtils::trim(trimmedExpr.substr(dotPos + 1)));
        } else {
            groupByColumns.push_back(trimmedExpr);
        }
    }
    
    return groupByColumns;
}

std::vector<std::pair<std::string, SortOrder>> ASTProcessor::parseOrderByClause(const std::string& orderByClause) {
    std::vector<std::pair<std::string, SortOrder>> orderByColumns;
    
    if (orderByClause.empty()) {
        return orderByColumns;
    }
    
    // Split by commas
    std::vector<std::string> columnExpressions = tokenizeClause(orderByClause, ",");
    
    for (const auto& expr : columnExpressions) {
        std::string trimmedExpr = StringUtils::trim(expr);
        SortOrder order = SortOrder::ASCENDING;  // Default order
        
        // Check for ASC/DESC suffix
        if (StringUtils::endsWith(StringUtils::toUpper(trimmedExpr), " ASC")) {
            trimmedExpr = StringUtils::trim(trimmedExpr.substr(0, trimmedExpr.length() - 4));
        } else if (StringUtils::endsWith(StringUtils::toUpper(trimmedExpr), " DESC")) {
            trimmedExpr = StringUtils::trim(trimmedExpr.substr(0, trimmedExpr.length() - 5));
            order = SortOrder::DESCENDING;
        }
        
        // Handle possible table prefix
        size_t dotPos = trimmedExpr.find('.');
        if (dotPos != std::string::npos) {
            // Extract column name only (ignore table prefix)
            orderByColumns.emplace_back(StringUtils::trim(trimmedExpr.substr(dotPos + 1)), order);
        } else {
            orderByColumns.emplace_back(trimmedExpr, order);
        }
    }
    
    return orderByColumns;
}

std::vector<std::string> ASTProcessor::tokenizeClause(const std::string& clause, const std::string& delimiter) {
    std::vector<std::string> tokens;
    
    if (clause.empty()) {
        return tokens;
    }
    
    // Simple tokenizer for comma-separated lists
    if (delimiter == ",") {
        std::string currentToken;
        bool inQuotes = false;
        char quoteChar = '\0';
        int parenthesisDepth = 0;
        
        for (char c : clause) {
            if ((c == '\'' || c == '"') && (quoteChar == '\0' || c == quoteChar)) {
                inQuotes = !inQuotes;
                if (inQuotes) quoteChar = c;
                else quoteChar = '\0';
                currentToken += c;
            } else if (c == '(' && !inQuotes) {
                parenthesisDepth++;
                currentToken += c;
            } else if (c == ')' && !inQuotes) {
                parenthesisDepth--;
                currentToken += c;
            } else if (c == ',' && !inQuotes && parenthesisDepth == 0) {
                tokens.push_back(StringUtils::trim(currentToken));
                currentToken.clear();
            } else {
                currentToken += c;
            }
        }
        
        if (!currentToken.empty()) {
            tokens.push_back(StringUtils::trim(currentToken));
        }
        
        return tokens;
    }
    
    // For other delimiters (like AND), use simple string splitting
    size_t pos = 0;
    std::string token;
    std::string uppercaseClause = StringUtils::toUpper(clause);
    std::string uppercaseDelimiter = StringUtils::toUpper(delimiter);
    
    while ((pos = uppercaseClause.find(uppercaseDelimiter, pos)) != std::string::npos) {
        token = StringUtils::trim(clause.substr(0, pos));
        tokens.push_back(token);
        pos += uppercaseDelimiter.length();
        clause = StringUtils::trim(clause.substr(pos));
        uppercaseClause = StringUtils::toUpper(clause);
        pos = 0;
    }
    
    // Add the remaining part
    if (!clause.empty()) {
        tokens.push_back(StringUtils::trim(clause));
    }
    
    return tokens;
}

std::string ASTProcessor::extractClause(
    const std::string& query, 
    const std::string& clauseStart, 
    const std::string& clauseEnd) {
    
    std::string uppercaseQuery = StringUtils::toUpper(query);
    std::string uppercaseStart = StringUtils::toUpper(clauseStart);
    
    size_t startPos = uppercaseQuery.find(uppercaseStart);
    if (startPos == std::string::npos) {
        return "";
    }
    
    // Move past the clause keyword
    startPos += uppercaseStart.length();
    
    // Find the end of the clause
    size_t endPos = std::string::npos;
    if (clauseEnd != "$") {
        // Split clauseEnd by "|" to handle multiple possible endings
        std::istringstream endStream(clauseEnd);
        std::string endOption;
        while (std::getline(endStream, endOption, '|')) {
            std::string uppercaseEnd = StringUtils::toUpper(endOption);
            size_t pos = uppercaseQuery.find(uppercaseEnd, startPos);
            if (pos != std::string::npos && (endPos == std::string::npos || pos < endPos)) {
                endPos = pos;
            }
        }
    }
    
    // Extract the clause content
    if (endPos == std::string::npos) {
        // If no end marker found, use the rest of the query
        return StringUtils::trim(query.substr(startPos));
    } else {
        return StringUtils::trim(query.substr(startPos, endPos - startPos));
    }
}

} // namespace SQLQueryProcessor