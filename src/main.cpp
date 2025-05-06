#include <iostream>
#include <cassert>
#include <memory>
#include <chrono>
#include "../include/Operations/Select.hpp"
#include "../include/DataHandling/Table.hpp"
#include "../include/DataHandling/Condition.hpp"
#include "../include/Operations/Filter.hpp"
#include "../include/Operations/OrderBy.hpp"
#include "../include/Operations/Aggregator.hpp"
#include "../include/Operations/Project.hpp"
#include "../include/Operations/Join.hpp"
#include "../include/SQLProcessing/SQLQueryProcessor.hpp"

#define USE_GPU 1

using namespace GPUDBMS;

// Helper function to create a test table
Table createTestTable()
{
    std::vector<Column> columns = {
        Column("id", DataType::INT),
        Column("name", DataType::VARCHAR),
        Column("age", DataType::INT),
        Column("salary", DataType::DOUBLE),
    };

    Table table(columns);

    // Add data to columns
    auto &idCol = static_cast<ColumnDataImpl<int> &>(table.getColumnData("id"));
    auto &nameCol = static_cast<ColumnDataImpl<std::string> &>(table.getColumnData("name"));
    auto &ageCol = static_cast<ColumnDataImpl<int> &>(table.getColumnData("age"));
    auto &salaryCol = static_cast<ColumnDataImpl<double> &>(table.getColumnData("salary"));

    // Add 5 rows
    for (int i = 1; i <= 10000000; i++)
    {
        idCol.append(i);
        nameCol.append("Person" + std::to_string(i));
        ageCol.append(i);
        salaryCol.append(i);

        // Finalize each row after adding all column values
        table.finalizeRow();
    }

    // // Add an additional row for testing aggregations
    // idCol.append(6);
    // nameCol.append("Person" + std::to_string(6));
    // ageCol.append(25);
    // salaryCol.append(50000.0 + 1 * 10000.0);
    // table.finalizeRow();

    // // Add an additional row for testing
    // idCol.append(7);
    // nameCol.append("Person" + std::to_string(7));
    // ageCol.append(35);
    // salaryCol.append(70000.0 + 2 * 10000.0);
    // table.finalizeRow();

    // // Add an additional row for testing
    // idCol.append(8);
    // nameCol.append("Person" + std::to_string(8));
    // ageCol.append(35);
    // salaryCol.append(90000.0 + 3 * 10000.0);
    // table.finalizeRow();

    return table;
}

// Test Select operation
void testSelect()
{
    std::cout << "Testing Select operation..." << std::endl;

    Table testTable = createTestTable();

    // Test simple selection (age > 30)
    auto condition = ConditionBuilder::lessThan("age", "5000000");
    Select selectOp(testTable, *condition);

    // Execute on CPU
    auto start = std::chrono::high_resolution_clock::now();
    Table resultCPU = selectOp.execute();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << resultCPU.getRowCount() << " rows selected on CPU" << std::endl;
    std::cout << "CPU Select execution time: " << elapsed.count() << " seconds" << std::endl;
    assert(resultCPU.getRowCount() == 4999999);
    std::cout << "CPU Select test passed!" << std::endl;

    // Execute on GPU if available
    try
    {
        auto start = std::chrono::high_resolution_clock::now();
        Table resultGPU = selectOp.execute(USE_GPU);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << resultGPU.getRowCount() << " rows selected on GPU" << std::endl;
        std::cout << "GPU Select execution time: " << elapsed.count() << " seconds" << std::endl;
        assert(resultGPU.getRowCount() == 4999999);
        std::cout << "GPU Select test passed!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "GPU execution not available: " << e.what() << std::endl;
    }
}

// Test Project operation
void testProject()
{
    std::cout << "Testing Project operation..." << std::endl;

    Table testTable = createTestTable();

    // Test projection (id, name)
    std::vector<std::string> columns = {"id", "name"};
    Project projectOp(testTable, columns);

    // Execute on CPU
    Table resultCPU = projectOp.executeCPU();
    assert(resultCPU.getColumnCount() == 2);
    assert(resultCPU.getColumnIndex("id") != -1);
    assert(resultCPU.getColumnIndex("name") != -1);
    assert(resultCPU.getColumnIndex("age") == -1);

    // Execute on GPU if available
    try
    {
        Table resultGPU = projectOp.execute();
        assert(resultGPU.getColumnCount() == 2);
        assert(resultGPU.getColumnIndex("id") != -1);
        assert(resultGPU.getColumnIndex("name") != -1);
        assert(resultGPU.getColumnIndex("age") == -1);
        std::cout << "GPU Project test passed!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "GPU execution not available: " << e.what() << std::endl;
    }

    std::cout << "CPU Project test passed!" << std::endl;
}

// Test complex condition
void testComplexCondition()
{
    std::cout << "Testing complex conditions..." << std::endl;

    Table testTable = createTestTable();

    auto ageCondition = ConditionBuilder::greaterThan("age", "1000000");
    auto ageCondition2 = ConditionBuilder::lessThan("age", "9000000");
    auto salaryCondition = ConditionBuilder::lessThan("salary", "10");

    auto andCondition = ConditionBuilder::And(std::move(ageCondition), std::move(ageCondition2));
    auto complexCondition = ConditionBuilder::Or(std::move(andCondition), std::move(salaryCondition));

    Select selectOp(testTable, *complexCondition);

    // Execute on CPU
    auto start = std::chrono::high_resolution_clock::now();
    Table resultCPU = selectOp.execute();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << resultCPU.getRowCount() << " rows selected on CPU" << std::endl;
    std::cout << "CPU Select execution time: " << elapsed.count() << " seconds" << std::endl;
    assert(resultCPU.getRowCount() == 8000008);
    std::cout << "CPU Select test passed!" << std::endl;

    // Execute on GPU if available
    try
    {
        auto start = std::chrono::high_resolution_clock::now();
        Table resultGPU = selectOp.execute(USE_GPU);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << resultGPU.getRowCount() << " rows selected on GPU" << std::endl;
        std::cout << "GPU Select execution time: " << elapsed.count() << " seconds" << std::endl;
        assert(resultGPU.getRowCount() == 8000008);
        std::cout << "GPU Select test passed!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "GPU execution not available: " << e.what() << std::endl;
    }
}

void testOrderBy()
{
    std::cout << "Testing OrderBy operation..." << std::endl;

    Table testTable = createTestTable();

    // Test sorting by age in descending order
    OrderBy orderByOp(testTable, "age", SortOrder::DESC);

    // Execute on CPU
    Table resultCPU = orderByOp.executeCPU();

    // Verify results (should be sorted by age in descending order)
    assert(resultCPU.getRowCount() == 5);
    assert(resultCPU.getIntValue(resultCPU.getColumnIndex("age"), 0) == 45); // Person5
    assert(resultCPU.getIntValue(resultCPU.getColumnIndex("age"), 1) == 40); // Person4
    assert(resultCPU.getIntValue(resultCPU.getColumnIndex("age"), 2) == 35); // Person3
    assert(resultCPU.getIntValue(resultCPU.getColumnIndex("age"), 3) == 30); // Person2
    assert(resultCPU.getIntValue(resultCPU.getColumnIndex("age"), 4) == 25); // Person1

    // Test sorting by name in ascending order
    OrderBy nameOrderByOp(testTable, "name", SortOrder::ASC);

    Table nameResultCPU = nameOrderByOp.executeCPU();

    // Verify results (should be sorted by name alphabetically)
    assert(nameResultCPU.getRowCount() == 5);
    assert(nameResultCPU.getStringValue(nameResultCPU.getColumnIndex("name"), 0) == "Person1");

    // Test multi-column sorting (sort by salary DESC, then by name ASC)
    std::vector<std::string> sortColumns = {"salary", "name"};
    std::vector<SortOrder> sortOrders = {SortOrder::DESC, SortOrder::ASC};
    OrderBy multiColOrderByOp(testTable, sortColumns, sortOrders);

    Table multiColResult = multiColOrderByOp.executeCPU();

    // Should be ordered by decreasing salary
    assert(multiColResult.getDoubleValue(multiColResult.getColumnIndex("salary"), 0) == 100000.0); // Person5
    assert(multiColResult.getDoubleValue(multiColResult.getColumnIndex("salary"), 1) == 90000.0);  // Person4

    // Execute on GPU if available
    try
    {
        Table resultGPU = orderByOp.execute();
        assert(resultGPU.getRowCount() == 5);
        assert(resultGPU.getIntValue(resultGPU.getColumnIndex("age"), 0) == 45);
        std::cout << "GPU OrderBy test passed!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "GPU execution not available: " << e.what() << std::endl;
    }

    std::cout << "CPU OrderBy test passed!" << std::endl;
}
void testAggregator()
{
    std::cout << "Testing Aggregator operation..." << std::endl;

    Table testTable = createTestTable();

    // Test COUNT on the entire table
    Aggregator countAggregator(testTable, AggregateFunction::COUNT, "id", std::nullopt, "total_count");
    Table countResult = countAggregator.executeCPU();

    // Verify results
    assert(countResult.getRowCount() == 1);
    assert(countResult.getColumnCount() == 1);
    assert(countResult.getIntValue(0, 0) == 5); // 5 rows in the test table

    // Test SUM and AVG on the salary column
    std::vector<Aggregation> aggregations = {
        Aggregation(AggregateFunction::SUM, "salary", "total_salary"),
        Aggregation(AggregateFunction::AVG, "salary", "avg_salary")};

    Aggregator multiAggregator(testTable, aggregations);
    Table multiResult = multiAggregator.executeCPU();

    // Verify results
    assert(multiResult.getRowCount() == 1);
    assert(multiResult.getColumnCount() == 2);

    // Check sum: 60000 + 70000 + 80000 + 90000 + 100000 = 400000
    assert(std::abs(multiResult.getDoubleValue(0, 0) - 400000.0) < 0.001);

    // Check avg: 400000 / 5 = 80000
    assert(std::abs(multiResult.getDoubleValue(1, 0) - 80000.0) < 0.001);

    // Test MIN and MAX
    aggregations = {
        Aggregation(AggregateFunction::MIN, "age", "min_age"),
        Aggregation(AggregateFunction::MAX, "age", "max_age")};

    Aggregator minMaxAggregator(testTable, aggregations);
    Table minMaxResult = minMaxAggregator.executeCPU();

    // Verify results
    assert(minMaxResult.getRowCount() == 1);
    assert(minMaxResult.getColumnCount() == 2);
    assert(minMaxResult.getIntValue(0, 0) == 25); // Min age
    assert(minMaxResult.getIntValue(1, 0) == 45); // Max age

    // Test GROUP BY
    aggregations = {
        Aggregation(AggregateFunction::COUNT, "id", "count"),
        Aggregation(AggregateFunction::AVG, "salary", "avg_salary")};

    // Group by even/odd id (which will create 2 groups)
    // We need to add a column for this
    auto testTableCopy = testTable;
    testTableCopy.addColumn(Column("id_group", DataType::INT));

    auto &idGroupCol = static_cast<ColumnDataImpl<int> &>(testTableCopy.getColumnData("id_group"));
    for (int i = 1; i <= 5; i++)
    {
        // Group 0 for odd ids, Group 1 for even ids
        idGroupCol.append(i % 2);
    }

    Aggregator groupByAggregator(testTableCopy, aggregations, "id_group");
    Table groupByResult = groupByAggregator.executeCPU();

    // Verify results
    assert(groupByResult.getRowCount() == 2);    // Two groups
    assert(groupByResult.getColumnCount() == 3); // Group column + 2 aggregations

    // Try on GPU if available
    try
    {
        Table resultGPU = countAggregator.execute();
        assert(resultGPU.getRowCount() == 1);
        assert(resultGPU.getIntValue(0, 0) == 5);
        std::cout << "GPU Aggregator test passed!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "GPU execution not available: " << e.what() << std::endl;
    }

    std::cout << "CPU Aggregator test passed!" << std::endl;
}
void testJoin()
{
    std::cout << "Testing Join operation..." << std::endl;

    // Create two test tables for joining
    Table leftTable = createTestTable(); // This is our existing test table

    // Create a second test table with ids that will match some from the first table
    std::vector<Column> rightColumns = {
        Column("id", DataType::INT),
        Column("department", DataType::VARCHAR),
        Column("location", DataType::VARCHAR)};

    Table rightTable(rightColumns);

    // Add data to the right table
    auto &rightIdCol = static_cast<ColumnDataImpl<int> &>(rightTable.getColumnData("id"));
    auto &rightDeptCol = static_cast<ColumnDataImpl<std::string> &>(rightTable.getColumnData("department"));
    auto &rightLocCol = static_cast<ColumnDataImpl<std::string> &>(rightTable.getColumnData("location"));

    // Add some rows that will match (id: 2, 4) and some that won't
    rightIdCol.append(2);
    rightDeptCol.append("Engineering");
    rightLocCol.append("Building A");

    rightIdCol.append(4);
    rightDeptCol.append("Marketing");
    rightLocCol.append("Building B");

    rightIdCol.append(6);
    rightDeptCol.append("Finance");
    rightLocCol.append("Building C");

    // Finalize rows
    rightTable.finalizeRow();
    rightTable.finalizeRow();
    rightTable.finalizeRow();

    // Create join condition: leftTable.id = rightTable.id
    auto joinCondition = ConditionBuilder::equals("id", "id");
    // Test INNER JOIN
    Join innerJoinOp(leftTable, rightTable, *joinCondition, JoinType::INNER);
    Table innerJoinResult = innerJoinOp.executeCPU();

    // Verify inner join results
    assert(innerJoinResult.getRowCount() == 2); // Should match 2 rows (id 2 and 4)
    assert(innerJoinResult.getColumnCount() == leftTable.getColumnCount() + rightTable.getColumnCount());

    // Verify some values from the joined rows
    int leftIdCol = innerJoinResult.getColumnIndex("id");

    // Check that the join contains the expected IDs
    std::vector<int> expectedIds = {2, 4}; // IDs that should be in the join result
    bool foundId2 = false;
    bool foundId4 = false;

    for (size_t row = 0; row < innerJoinResult.getRowCount(); ++row)
    {
        int rowId = innerJoinResult.getIntValue(leftIdCol, row);
        if (rowId == 2)
            foundId2 = true;
        if (rowId == 4)
            foundId4 = true;
    }

    assert(foundId2 && foundId4);

    // Test LEFT JOIN
    Join leftJoinOp(leftTable, rightTable, *joinCondition, JoinType::LEFT);
    Table leftJoinResult = leftJoinOp.executeCPU();

    // Verify left join results
    assert(leftJoinResult.getRowCount() == 5); // Should include all rows from left table

    // Try on GPU if available
    try
    {
        Table resultGPU = innerJoinOp.execute();
        assert(resultGPU.getRowCount() == 2);
        std::cout << "GPU Join test passed!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "GPU execution not available: " << e.what() << std::endl;
    }

    std::cout << "CPU Join test passed!" << std::endl;
}

void testSelectQueryParser(SQLQueryProcessor &processor)
{

    try
    {
        std::cout << "Running query: SELECT id, name FROM employees WHERE age > 30" << std::endl;
        Table result = processor.processQuery("SELECT id, name FROM employees WHERE age > 30");
        std::cout << "Query execution successful. Result rows: " << result.getRowCount() << std::endl;

        // Print the result for debugging
        std::cout << "Query result:" << std::endl;
        for (size_t row = 0; row < result.getRowCount(); ++row)
        {
            std::cout << "Row " << row << ": ";
            std::cout << "id=" << result.getIntValue(0, row) << ", ";
            std::cout << "name=" << result.getStringValue(1, row) << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error executing query: " << e.what() << std::endl;
        throw; // Re-throw to fail the test
    }
}

void testAggregationsParser(SQLQueryProcessor &processor)
{
    try
    {
        std::cout << "Running aggregation query: SELECT MAX(salary) as max_salary, MIN(age) as min_age FROM employees" << std::endl;
        Table aggResult = processor.processQuery("SELECT MAX(salary) as max_salary, MIN(age) as min_age FROM employees");
        std::cout << "Aggregation query successful. Result rows: " << aggResult.getRowCount() << std::endl;

        // Print the result
        std::cout << "Aggregation result:" << std::endl;
        std::cout << "max_salary=" << aggResult.getDoubleValue(0, 0) << ", ";
        std::cout << "min_age=" << aggResult.getIntValue(1, 0) << std::endl;

        // Test GROUP BY query
        std::cout << "Running GROUP BY query: SELECT age, COUNT(id) as count FROM employees GROUP BY age" << std::endl;
        Table groupResult = processor.processQuery("SELECT age, COUNT(id) as count FROM employees GROUP BY age");
        std::cout << "GROUP BY query successful. Result rows: " << groupResult.getRowCount() << std::endl;

        // Print the result
        std::cout << "GROUP BY result:" << std::endl;
        for (size_t row = 0; row < groupResult.getRowCount(); ++row)
        {
            std::cout << "age=" << groupResult.getIntValue(0, row) << ", ";
            std::cout << "count=" << groupResult.getIntValue(1, row) << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error executing aggregation query: " << e.what() << std::endl;
        throw; // Re-throw to fail the test
    }
}

void testOrderBySQL(SQLQueryProcessor &processor)
{
    std::cout << "Testing SQL ORDER BY..." << std::endl;

    // Single column ORDER BY
    std::cout << "Running query: SELECT * FROM employees ORDER BY age DESC" << std::endl;
    Table result1 = processor.processQuery("SELECT * FROM employees ORDER BY age DESC");

    // Print results
    std::cout << "Results ordered by age DESC:" << std::endl;
    for (size_t row = 0; row < result1.getRowCount(); ++row)
    {
        std::cout << "Row " << row << ": ";
        std::cout << "id=" << result1.getIntValue(0, row) << ", ";
        std::cout << "name=" << result1.getStringValue(1, row) << ", ";
        std::cout << "age=" << result1.getIntValue(2, row) << ", ";
        std::cout << "salary=" << result1.getDoubleValue(3, row) << std::endl;
    }

    // Multi-column ORDER BY
    std::cout << "Running query: SELECT * FROM employees ORDER BY age ASC, salary DESC" << std::endl;
    Table result2 = processor.processQuery("SELECT * FROM employees ORDER BY age ASC, salary DESC");

    // Print results
    std::cout << "Results ordered by age ASC, salary DESC:" << std::endl;
    for (size_t row = 0; row < result2.getRowCount(); ++row)
    {
        std::cout << "Row " << row << ": ";
        std::cout << "id=" << result2.getIntValue(0, row) << ", ";
        std::cout << "name=" << result2.getStringValue(1, row) << ", ";
        std::cout << "age=" << result2.getIntValue(2, row) << ", ";
        std::cout << "salary=" << result2.getDoubleValue(3, row) << std::endl;
    }
}

void testSQLQueryProcessor()
{
    std::cout << "Testing SQL Query Processor..." << std::endl;

    SQLQueryProcessor processor;

    // Create a test table and register it with the processor
    Table employeesTable = createTestTable();
    std::cout << "Created employees table with " << employeesTable.getRowCount() << " rows" << std::endl;

    // Print the content of the test table for debugging
    std::cout << "Employees table content:" << std::endl;
    for (size_t row = 0; row < employeesTable.getRowCount(); ++row)
    {
        std::cout << "Row " << row << ": ";
        std::cout << "id=" << employeesTable.getIntValue(0, row) << ", ";
        std::cout << "name=" << employeesTable.getStringValue(1, row) << ", ";
        std::cout << "age=" << employeesTable.getIntValue(2, row) << ", ";
        std::cout << "salary=" << employeesTable.getDoubleValue(3, row) << std::endl;
    }

    processor.registerTable("employees", employeesTable);

    testSelectQueryParser(processor);
    testAggregationsParser(processor);
    testOrderBySQL(processor);

    std::cout << "SQL Query Processor test passed!" << std::endl;
}

void testCSVLoading()
{
    std::cout << "Testing CSV loading functionality with annotated headers..." << std::endl;

    // try
    // {
    //     // Initialize SQLQueryProcessor with the data directory
    //     SQLQueryProcessor processor("/mnt/g/MyRepos/SQLQueryProcessor/data");

    //     // Print the Students table schema and data with detailed type information
    //     Table studentsTable = processor.getTable("Students");
        
    //     std::cout << "\n--- Students Table Schema (AFTER FIX) ---\n";
    //     for (size_t i = 0; i < studentsTable.getColumnCount(); i++) {
    //         std::cout << "Column " << i << ": " << studentsTable.getColumnName(i) 
    //                   << " (Type: " << static_cast<int>(studentsTable.getColumnType(i)) << ")" << std::endl;
    //     }
        
    //     // Now run a GPA query that should work properly
    //     std::cout << "\n--- Testing SELECT Query ---\n";
    //     std::cout << "Running query: SELECT * FROM Students WHERE gpa > 3.5" << std::endl;
    //     Table result = processor.processQueryAndSave("SELECT * FROM Students WHERE gpa > 3.5", "outputMah.csv");
        
    //     std::cout << "Query result has " << result.getRowCount() << " rows" << std::endl;
        
    //     // Print all matching students
    //     for (size_t i = 0; i < result.getRowCount(); i++) {
    //         std::cout << "Student: " << result.getStringValue(1, i) 
    //                   << ", GPA: " << result.getDoubleValue(3, i) << std::endl;
    //     }
    // }
    // catch (const std::exception &e)
    // {
    //     std::cerr << "Error in CSV test: " << e.what() << std::endl;
    // }
}

void testDateTimeSupport()
{
    std::cout << "Testing DateTime support..." << std::endl;
    
    // Create a table with a DateTime column
    std::vector<Column> columns = {
        Column("id", DataType::INT),
        Column("name", DataType::VARCHAR),
        Column("created_at", DataType::DATETIME)
    };
    
    Table table(columns);
    
    // Add sample data
    table.appendIntValue(0, 1);
    table.appendStringValue(1, "Event 1");
    table.appendStringValue(2, "2023-01-15 10:30:00");
    table.finalizeRow();
    
    table.appendIntValue(0, 2);
    table.appendStringValue(1, "Event 2");
    table.appendStringValue(2, "2023-02-20 14:45:30");
    table.finalizeRow();
    
    table.appendIntValue(0, 3);
    table.appendStringValue(1, "Event 3");
    table.appendStringValue(2, "2023-03-25 08:15:45");
    table.finalizeRow();
    
    // Test condition with DateTime
    auto condition = ConditionBuilder::greaterThan("created_at", "2023-02-01 00:00:00");
    Select selectOp(table, *condition);
    Table result = selectOp.execute(false); // Use CPU implementation
    
    std::cout << "Events after 2023-02-01: " << result.getRowCount() << std::endl;
    for (size_t i = 0; i < result.getRowCount(); i++) {
        std::cout << "Event: " << result.getStringValue(1, i) 
                  << ", Created: " << result.getStringValue(2, i) << std::endl;
    }
    
    std::cout << "DateTime support test completed!" << std::endl;
}
int main()
{
    try
    {
        // testSelect();
        // testProject();
        testComplexCondition();
        // testFilter();
        // testOrderBy();
        // testAggregator();
        // testJoin();
        // testSQLQueryProcessor();
        // testCSVLoading();
        // testDateTimeSupport();

        std::cout << "All tests passed successfully!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}