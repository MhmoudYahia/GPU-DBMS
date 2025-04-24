#include "Utilities/StringUtils.hpp"
#include <algorithm>
#include <random>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace SQLQueryProcessor {
namespace StringUtils {

std::string ltrim(const std::string& s) {
    std::string result = s;
    result.erase(result.begin(), std::find_if(result.begin(), result.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    return result;
}

std::string rtrim(const std::string& s) {
    std::string result = s;
    result.erase(std::find_if(result.rbegin(), result.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), result.end());
    return result;
}

std::string trim(const std::string& s) {
    return rtrim(ltrim(s));
}

std::string toUpper(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), 
                  [](unsigned char c) { return std::toupper(c); });
    return result;
}

std::string toLower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), 
                  [](unsigned char c) { return std::tolower(c); });
    return result;
}

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}

std::vector<std::string> splitAndTrim(const std::string& s, char delimiter) {
    std::vector<std::string> tokens = split(s, delimiter);
    for (auto& token : tokens) {
        token = trim(token);
    }
    return tokens;
}

std::string join(const std::vector<std::string>& strings, const std::string& delimiter) {
    if (strings.empty()) {
        return "";
    }
    
    std::ostringstream oss;
    oss << strings[0];
    
    for (size_t i = 1; i < strings.size(); ++i) {
        oss << delimiter << strings[i];
    }
    
    return oss.str();
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

bool startsWith(const std::string& str, const std::string& prefix) {
    if (str.length() < prefix.length()) {
        return false;
    }
    return str.compare(0, prefix.length(), prefix) == 0;
}

bool endsWith(const std::string& str, const std::string& suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

std::string replaceAll(const std::string& str, const std::string& from, const std::string& to) {
    if (from.empty()) {
        return str;
    }
    
    std::string result = str;
    size_t pos = 0;
    
    while ((pos = result.find(from, pos)) != std::string::npos) {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }
    
    return result;
}

bool tryParseInt(const std::string& s, int& result) {
    try {
        result = std::stoi(trim(s));
        return true;
    } catch (...) {
        return false;
    }
}

bool tryParseDouble(const std::string& s, double& result) {
    try {
        result = std::stod(trim(s));
        return true;
    } catch (...) {
        return false;
    }
}

std::string formatDouble(double value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

std::string removeQuotes(const std::string& s) {
    std::string result = trim(s);
    
    if (result.length() >= 2 && 
        ((result.front() == '"' && result.back() == '"') || 
         (result.front() == '\'' && result.back() == '\''))) {
        return result.substr(1, result.length() - 2);
    }
    
    return result;
}

std::string extractQuotedString(const std::string& s, size_t startPos, size_t& endPos) {
    endPos = startPos;
    if (startPos >= s.length()) {
        return "";
    }
    
    char quoteChar = s[startPos];
    if (quoteChar != '"' && quoteChar != '\'') {
        return "";
    }
    
    std::string result;
    bool escaped = false;
    
    for (size_t i = startPos + 1; i < s.length(); ++i) {
        char c = s[i];
        
        if (escaped) {
            // Handle escaped character
            result += c;
            escaped = false;
        } else if (c == '\\') {
            // Start of escape sequence
            escaped = true;
        } else if (c == quoteChar) {
            // End of quoted string
            endPos = i;
            return result;
        } else {
            // Regular character
            result += c;
        }
    }
    
    // If we got here, the string wasn't closed
    endPos = s.length() - 1;
    return result;
}

std::string generateUniqueId(const std::string& prefix) {
    static std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    static std::uniform_int_distribution<> dis(0, 35);
    
    std::string chars = "0123456789abcdefghijklmnopqrstuvwxyz";
    std::string id = prefix;
    
    for (int i = 0; i < 8; ++i) {
        id += chars[dis(gen)];
    }
    
    return id;
}

} // namespace StringUtils
} // namespace SQLQueryProcessor