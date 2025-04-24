#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace SQLQueryProcessor {

class InputParser {
public:
    InputParser() = default;
    ~InputParser() = default;
    
    // Parse command line arguments
    void parseArgs(int argc, char* argv[]);
    
    // Check if a specific flag was provided
    bool hasFlag(const std::string& flag) const;
    
    // Get value for a specific option
    std::string getOption(const std::string& option) const;
    
    // Get positional arguments
    const std::vector<std::string>& getPositionalArgs() const;
    
    // Process input from console
    std::string normalizeQuery(const std::string& query) const;
    
    // Check if a string is a valid SQL query
    bool isValidQuery(const std::string& query) const;
    
    // Extract command from input (for special commands like .help, .exit)
    bool isCommand(const std::string& input, std::string& command, std::string& args) const;
    
    // Read queries from a file
    std::vector<std::string> readQueriesFromFile(const std::string& filePath) const;
    
private:
    std::unordered_map<std::string, bool> flags;
    std::unordered_map<std::string, std::string> options;
    std::vector<std::string> positionalArgs;
    
    // Helper methods
    std::string trimWhitespace(const std::string& str) const;
    bool startsWithCommand(const std::string& input) const;
};

} // namespace SQLQueryProcessor