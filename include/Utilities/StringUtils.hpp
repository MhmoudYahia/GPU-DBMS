#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace SQLQueryProcessor
{
    namespace StringUtils
    {

        // String trimming functions
        std::string ltrim(const std::string &s);
        std::string rtrim(const std::string &s);
        std::string trim(const std::string &s);

        // String case conversion
        std::string toUpper(const std::string &s);
        std::string toLower(const std::string &s);

        // String splitting
        std::vector<std::string> split(const std::string &s, char delimiter);
        std::vector<std::string> splitAndTrim(const std::string &s, char delimiter);

        // String joining
        std::string join(const std::vector<std::string> &strings, const std::string &delimiter);

        // Check if string contains a substring
        bool contains(const std::string &haystack, const std::string &needle);

        // Check if string starts with prefix
        bool startsWith(const std::string &str, const std::string &prefix);

        // Check if string ends with suffix
        bool endsWith(const std::string &str, const std::string &suffix);

        // Replace all occurrences of a substring
        std::string replaceAll(const std::string &str, const std::string &from, const std::string &to);

        // Parse string to number safely
        bool tryParseInt(const std::string &s, int &result);
        bool tryParseDouble(const std::string &s, double &result);

        // Format a number with specific precision
        std::string formatDouble(double value, int precision);

        // Remove quotes from a string
        std::string removeQuotes(const std::string &s);

        // Extract quoted string
        std::string extractQuotedString(const std::string &s, size_t startPos, size_t &endPos);

        // Generate a unique identifier
        std::string generateUniqueId(const std::string &prefix = "");

    } // namespace StringUtils
} // namespace SQLQueryProcessor