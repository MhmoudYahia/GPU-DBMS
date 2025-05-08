#include "../../include/CLI/CommandLineInterface.hpp"
#include "../../include/SQLProcessing/SQLQueryProcessor.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <readline/readline.h>
#include <readline/history.h>

namespace GPUDBMS
{

    CommandLineInterface::CommandLineInterface(const std::string &dataDirectory)
        : m_quit(false), m_useGPU(true)
    {

        // Initialize the SQL Query Processor with the data directory
        m_processor = std::make_unique<SQLQueryProcessor>(dataDirectory);

        // Load available tables
        try
        {
            m_availableTables = m_processor->getTableNames();
        }
        catch (const std::exception &e)
        {
            std::cerr << "Warning: Could not load table list: " << e.what() << std::endl;
        }
    }

    CommandLineInterface::~CommandLineInterface()
    {
        // Clean up readline history
        clear_history();
    }

    void CommandLineInterface::run()
    {
        printWelcomeMessage();

        char *input;
        while (!m_quit)
        {
            // Use readline for input with command history
            input = readline("SQL> ");

            // Exit if EOF (Ctrl+D)
            if (input == nullptr)
            {
                std::cout << "Exiting..." << std::endl;
                break;
            }

            std::string command(input);

            // Add to history if not empty
            if (!command.empty())
            {
                add_history(input);
                processCommand(command);
            }

            // Free memory allocated by readline
            free(input);
        }
    }

    void CommandLineInterface::printWelcomeMessage()
    {
        std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                SQL Query Processor CLI                     ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
        std::cout << "Type SQL commands or special commands (start with \\)" << std::endl;
        std::cout << "Special commands:" << std::endl;
        std::cout << "  \\help                - Display this help message" << std::endl;
        std::cout << "  \\quit or \\exit       - Exit the program" << std::endl;
        std::cout << "  \\tables              - List available tables" << std::endl;
        std::cout << "  \\describe [table]    - Show table schema" << std::endl;
        std::cout << "  \\gpu [on|off]        - Enable/disable GPU execution" << std::endl;
        std::cout << "  \\save [filename]     - Save last query result to CSV" << std::endl;
        std::cout << "  \\timing [on|off]     - Enable/disable query timing" << std::endl;
        std::cout << "  \\load [filename]     - Load and execute SQL from file" << std::endl;
        std::cout << std::endl;
    }

    void CommandLineInterface::processCommand(const std::string &command)
    {
        // Trim whitespace from the command
        std::string trimmedCommand = trimString(command);

        // Skip empty commands
        if (trimmedCommand.empty())
        {
            return;
        }

        // Check if it's a special command (starts with \)
        if (trimmedCommand[0] == '\\')
        {
            processSpecialCommand(trimmedCommand);
        }
        else
        {
            // Process as SQL query
            executeQuery(trimmedCommand);
        }
    }

    void CommandLineInterface::processSpecialCommand(const std::string &command)
    {
        // Extract the command name and arguments
        std::string cmdName = command.substr(1); // Skip the leading '\'
        std::string argument;

        // Split into command and arguments
        size_t spacePos = cmdName.find(' ');
        if (spacePos != std::string::npos)
        {
            argument = trimString(cmdName.substr(spacePos + 1));
            cmdName = cmdName.substr(0, spacePos);
        }

        // Convert command to lowercase for case-insensitive comparison
        std::transform(cmdName.begin(), cmdName.end(), cmdName.begin(), ::tolower);

        // Process the command
        if (cmdName == "help" || cmdName == "h")
        {
            printWelcomeMessage();
        }
        else if (cmdName == "quit" || cmdName == "exit" || cmdName == "q")
        {
            m_quit = true;
            std::cout << "Exiting SQL Query Processor CLI. Goodbye!" << std::endl;
        }
        else if (cmdName == "tables" || cmdName == "t")
        {
            showTables();
        }
        else if (cmdName == "describe" || cmdName == "desc" || cmdName == "d")
        {
            if (argument.empty())
            {
                std::cout << "Error: Table name required for \\describe command" << std::endl;
            }
            else
            {
                describeTable(argument);
            }
        }
        else if (cmdName == "gpu")
        {
            if (argument == "on")
            {
                m_useGPU = true;
                std::cout << "GPU execution enabled" << std::endl;
            }
            else if (argument == "off")
            {
                m_useGPU = false;
                std::cout << "GPU execution disabled (using CPU)" << std::endl;
            }
            else
            {
                std::cout << "GPU execution is currently " << (m_useGPU ? "enabled" : "disabled") << std::endl;
            }
        }
        else if (cmdName == "save")
        {
            if (argument.empty())
            {
                std::cout << "Error: Filename required for \\save command" << std::endl;
            }
            else if (m_lastResult.getRowCount() == 0)
            {
                std::cout << "Error: No query result to save" << std::endl;
            }
            else
            {
                saveResultToCSV(argument);
            }
        }
        else if (cmdName == "timing")
        {
            if (argument == "on")
            {
                m_showTiming = true;
                std::cout << "Query timing enabled" << std::endl;
            }
            else if (argument == "off")
            {
                m_showTiming = false;
                std::cout << "Query timing disabled" << std::endl;
            }
            else
            {
                std::cout << "Query timing is currently " << (m_showTiming ? "enabled" : "disabled") << std::endl;
            }
        }
        else if (cmdName == "load")
        {
            if (argument.empty())
            {
                std::cout << "Error: Filename required for \\load command" << std::endl;
            }
            else
            {
                loadAndExecuteSQL(argument);
            }
        }
        else
        {
            std::cout << "Error: Unknown command '" << cmdName << "'" << std::endl;
        }
    }

    void CommandLineInterface::executeQuery(const std::string &query)
    {
        try
        {
            // Record start time for timing
            auto startTime = std::chrono::high_resolution_clock::now();

            // Execute the query
            Table result = m_processor->processQuery(query, m_useGPU);

            // Record end time
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

            // Store the result for potential saving
            m_lastResult = result;

            // Display the result
            displayResultTable(result);

            // Show timing if enabled
            if (m_showTiming)
            {
                std::cout << "Query executed in " << duration.count() << " ms" << std::endl;
            }

            std::cout << result.getRowCount() << " row(s) returned" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error executing query: " << e.what() << std::endl;
        }
    }

    void CommandLineInterface::displayResultTable(const Table &table)
    {
        ResultDisplayer displayer;
        displayer.displayTable(table);
    }

    void CommandLineInterface::showTables()
    {
        // Refresh the list of available tables
        try
        {
            m_availableTables = m_processor->getTableNames();
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error retrieving table list: " << e.what() << std::endl;
            return;
        }

        if (m_availableTables.empty())
        {
            std::cout << "No tables available" << std::endl;
            return;
        }

        std::cout << "Available tables:" << std::endl;
        for (const auto &tableName : m_availableTables)
        {
            std::cout << "  " << tableName << std::endl;
        }
    }

    void CommandLineInterface::describeTable(const std::string &tableName)
    {
        try
        {
            Table table = m_processor->getTable(tableName);

            std::cout << "Table: " << tableName << std::endl;
            std::cout << "Columns:" << std::endl;

            const auto &columns = table.getColumns();
            for (size_t i = 0; i < columns.size(); ++i)
            {
                std::cout << "  " << columns[i].getName() << " (";

                // Output the data type
                switch (columns[i].getType())
                {
                case DataType::INT:
                    std::cout << "INT";
                    break;
                case DataType::FLOAT:
                    std::cout << "FLOAT";
                    break;
                case DataType::DOUBLE:
                    std::cout << "DOUBLE";
                    break;
                case DataType::VARCHAR:
                    std::cout << "VARCHAR";
                    break;
                case DataType::STRING:
                    std::cout << "STRING";
                    break;
                case DataType::BOOL:
                    std::cout << "BOOL";
                    break;
                case DataType::DATE:
                    std::cout << "DATE";
                    break;
                case DataType::DATETIME:
                    std::cout << "DATETIME";
                    break;
                default:
                    std::cout << "UNKNOWN";
                    break;
                }

                std::cout << ")" << std::endl;
            }

            std::cout << "Row count: " << table.getRowCount() << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error describing table: " << e.what() << std::endl;
        }
    }

    void CommandLineInterface::saveResultToCSV(const std::string &filename)
    {
        try
        {
            m_processor->saveQueryResultToCSV(m_lastResult, filename);
            std::cout << "Result saved to " << filename << ".csv" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error saving result: " << e.what() << std::endl;
        }
    }

    void CommandLineInterface::loadAndExecuteSQL(const std::string &filename)
    {
        try
        {
            std::ifstream file(filename);
            if (!file.is_open())
            {
                throw std::runtime_error("Could not open file: " + filename);
            }

            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string sqlQuery = buffer.str();

            // Execute the query from the file
            executeQuery(sqlQuery);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error loading SQL from file: " << e.what() << std::endl;
        }
    }

    std::string CommandLineInterface::trimString(const std::string &str)
    {
        // Find the first non-whitespace character
        size_t start = str.find_first_not_of(" \t\r\n");
        if (start == std::string::npos)
        {
            return ""; // String is all whitespace
        }

        // Find the last non-whitespace character
        size_t end = str.find_last_not_of(" \t\r\n");

        // Return the trimmed substring
        return str.substr(start, end - start + 1);
    }

} // namespace GPUDBMS