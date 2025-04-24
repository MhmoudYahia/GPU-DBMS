#include "Utilities/Logger.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <filesystem>

namespace SQLQueryProcessor {

// Initialize static members
std::atomic<bool> Logger::initialized(false);
std::string Logger::logFilePath;
std::atomic<LogLevel> Logger::minLogLevel(LogLevel::INFO);
std::atomic<bool> Logger::consoleOutputEnabled(true);
std::atomic<bool> Logger::fileOutputEnabled(true);
std::atomic<size_t> Logger::maxFileSizeMB(10); // Default: 10MB
std::atomic<size_t> Logger::maxBackupCount(3); // Default: 3 backup files
std::ofstream Logger::logFile;
std::mutex Logger::logMutex;

void Logger::init(const std::string& logFileName) {
    std::lock_guard<std::mutex> lock(logMutex);
    
    if (initialized) {
        // Close existing log file if open
        if (logFile.is_open()) {
            logFile.close();
        }
    }
    
    logFilePath = logFileName;
    
    // Create directory if it doesn't exist
    std::filesystem::path path(logFilePath);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    
    // Open log file
    logFile.open(logFilePath, std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Failed to open log file: " << logFilePath << std::endl;
        fileOutputEnabled = false;
    }
    
    initialized = true;
    
    // Log initialization message
    log(LogLevel::INFO, "Logger initialized");
}

void Logger::log(LogLevel level, const std::string& message) {
    if (level < minLogLevel) {
        return;
    }
    
    std::string formattedMessage = formatLogMessage(level, message);
    
    std::lock_guard<std::mutex> lock(logMutex);
    
    // Output to console if enabled
    if (consoleOutputEnabled) {
        if (level == LogLevel::ERROR || level == LogLevel::CRITICAL) {
            std::cerr << formattedMessage << std::endl;
        } else {
            std::cout << formattedMessage << std::endl;
        }
    }
    
    // Output to file if enabled
    if (fileOutputEnabled && logFile.is_open()) {
        logFile << formattedMessage << std::endl;
        logFile.flush();
        
        // Check if log rotation is needed
        checkRotateLogFile();
    }
}

void Logger::debug(const std::string& message) {
    log(LogLevel::DEBUG, message);
}

void Logger::info(const std::string& message) {
    log(LogLevel::INFO, message);
}

void Logger::warning(const std::string& message) {
    log(LogLevel::WARNING, message);
}

void Logger::error(const std::string& message) {
    log(LogLevel::ERROR, message);
}

void Logger::critical(const std::string& message) {
    log(LogLevel::CRITICAL, message);
}

void Logger::setLogLevel(LogLevel level) {
    minLogLevel = level;
}

void Logger::enableConsoleOutput(bool enable) {
    consoleOutputEnabled = enable;
}

void Logger::enableFileOutput(bool enable) {
    fileOutputEnabled = enable;
}

void Logger::setMaxFileSize(size_t maxSizeMB) {
    maxFileSizeMB = maxSizeMB;
}

void Logger::setMaxBackupCount(size_t count) {
    maxBackupCount = count;
}

bool Logger::isInitialized() {
    return initialized;
}

LogLevel Logger::getLogLevel() {
    return minLogLevel;
}

void Logger::shutdown() {
    std::lock_guard<std::mutex> lock(logMutex);
    
    if (logFile.is_open()) {
        log(LogLevel::INFO, "Logger shutting down");
        logFile.close();
    }
    
    initialized = false;
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:    return "DEBUG";
        case LogLevel::INFO:     return "INFO";
        case LogLevel::WARNING:  return "WARNING";
        case LogLevel::ERROR:    return "ERROR";
        case LogLevel::CRITICAL: return "CRITICAL";
        default:                 return "UNKNOWN";
    }
}

std::string Logger::formatLogMessage(LogLevel level, const std::string& message) {
    std::ostringstream oss;
    oss << getCurrentTimestamp() << " [" << levelToString(level) << "] " << message;
    return oss.str();
}

std::string Logger::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::tm tm_buf;
    #ifdef _WIN32
        localtime_s(&tm_buf, &now_time_t);
    #else
        localtime_r(&now_time_t, &tm_buf);
    #endif
    
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << now_ms.count();
    
    return oss.str();
}

void Logger::checkRotateLogFile() {
    if (!fileOutputEnabled || !logFile.is_open()) {
        return;
    }
    
    // Get current file size
    logFile.flush();
    size_t currentSizeBytes = 0;
    
    try {
        currentSizeBytes = std::filesystem::file_size(logFilePath);
    } catch (const std::exception& e) {
        std::cerr << "Error checking log file size: " << e.what() << std::endl;
        return;
    }
    
    // Convert max size to bytes
    size_t maxSizeBytes = maxFileSizeMB * 1024 * 1024;
    
    if (currentSizeBytes >= maxSizeBytes) {
        // Close current log file
        logFile.close();
        
        // Rotate log files
        rotateLogFiles();
        
        // Open new log file
        logFile.open(logFilePath, std::ios::app);
        if (!logFile.is_open()) {
            std::cerr << "Failed to reopen log file after rotation" << std::endl;
            fileOutputEnabled = false;
        }
    }
}

void Logger::rotateLogFiles() {
    // Delete oldest backup if it exists
    std::string oldestBackup = logFilePath + "." + std::to_string(maxBackupCount);
    try {
        std::filesystem::remove(oldestBackup);
    } catch (...) {
        // Ignore errors if file doesn't exist
    }
    
    // Shift existing backups
    for (int i = maxBackupCount - 1; i >= 1; --i) {
        std::string currentBackup = logFilePath + "." + std::to_string(i);
        std::string newBackup = logFilePath + "." + std::to_string(i + 1);
        
        try {
            if (std::filesystem::exists(currentBackup)) {
                std::filesystem::rename(currentBackup, newBackup);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error rotating log file " << currentBackup << ": " << e.what() << std::endl;
        }
    }
    
    // Rename current log file
    try {
        std::filesystem::rename(logFilePath, logFilePath + ".1");
    } catch (const std::exception& e) {
        std::cerr << "Error renaming current log file: " << e.what() << std::endl;
    }
}

} // namespace SQLQueryProcessor