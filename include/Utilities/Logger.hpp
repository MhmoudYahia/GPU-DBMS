#pragma once
#include <string>
#include <fstream>
#include <mutex>
#include <atomic>
#include <vector>
#include <memory>

namespace SQLQueryProcessor {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

class Logger {
public:
    // Initialize the logger with a log file
    static void init(const std::string& logFile);
    
    // Log a message with a specific level
    static void log(LogLevel level, const std::string& message);
    
    // Convenience methods for different log levels
    static void debug(const std::string& message);
    static void info(const std::string& message);
    static void warning(const std::string& message);
    static void error(const std::string& message);
    static void critical(const std::string& message);
    
    // Set the minimum log level
    static void setLogLevel(LogLevel level);
    
    // Enable/disable console output
    static void enableConsoleOutput(bool enable);
    
    // Enable/disable file output
    static void enableFileOutput(bool enable);
    
    // Set maximum log file size (in MB) and number of backups
    static void setMaxFileSize(size_t maxSizeMB);
    static void setMaxBackupCount(size_t count);
    
    // Get logger status
    static bool isInitialized();
    static LogLevel getLogLevel();
    
    // Clean up resources (called automatically on program exit)
    static void shutdown();
    
private:
    static std::atomic<bool> initialized;
    static std::string logFilePath;
    static std::atomic<LogLevel> minLogLevel;
    static std::atomic<bool> consoleOutputEnabled;
    static std::atomic<bool> fileOutputEnabled;
    static std::atomic<size_t> maxFileSizeMB;
    static std::atomic<size_t> maxBackupCount;
    static std::ofstream logFile;
    static std::mutex logMutex;
    
    // Private helper methods
    static std::string levelToString(LogLevel level);
    static std::string formatLogMessage(LogLevel level, const std::string& message);
    static std::string getCurrentTimestamp();
    static void checkRotateLogFile();
    static void rotateLogFiles();
    
    // Singleton protection
    Logger() = delete;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
};

// Convenience macro for logging with file and line info
#define LOG_DEBUG(message) \
    SQLQueryProcessor::Logger::debug(std::string(message) + " [" + __FILE__ + ":" + std::to_string(__LINE__) + "]")

#define LOG_INFO(message) \
    SQLQueryProcessor::Logger::info(message)

#define LOG_WARNING(message) \
    SQLQueryProcessor::Logger::warning(std::string(message) + " [" + __FILE__ + ":" + std::to_string(__LINE__) + "]")

#define LOG_ERROR(message) \
    SQLQueryProcessor::Logger::error(std::string(message) + " [" + __FILE__ + ":" + std::to_string(__LINE__) + "]")

#define LOG_CRITICAL(message) \
    SQLQueryProcessor::Logger::critical(std::string(message) + " [" + __FILE__ + ":" + std::to_string(__LINE__) + "]")

} // namespace SQLQueryProcessor