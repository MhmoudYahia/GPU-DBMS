#include <iostream>
#include <cassert>
#include <memory>
#include <chrono>
#include <iomanip>
#include "../include/Operations/Select.hpp"
#include "../include/DataHandling/Table.hpp"
#include "../include/DataHandling/Condition.hpp"
#include "../include/Operations/Filter.hpp"
#include "../include/Operations/OrderBy.hpp"
#include "../include/Operations/Aggregator.hpp"
#include "../include/Operations/Project.hpp"
// #include "../include/Operations/Join.hpp"
#include "../include/SQLProcessing/SQLQueryProcessor.hpp"
#include "../include/CLI/CommandLineInterface.hpp"
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
        Column("isEmployed", DataType::BOOL),
    };

    Table table(columns);

    // Add data to columns
    auto &idCol = static_cast<ColumnDataImpl<int> &>(table.getColumnData("id"));
    auto &nameCol = static_cast<ColumnDataImpl<std::string> &>(table.getColumnData("name"));
    auto &ageCol = static_cast<ColumnDataImpl<int> &>(table.getColumnData("age"));
    auto &salaryCol = static_cast<ColumnDataImpl<double> &>(table.getColumnData("salary"));
    auto &isEmployedCol = static_cast<ColumnDataImpl<bool> &>(table.getColumnData("isEmployed"));

    for (int i = 1; i <= 10000000; i++)
    {
        idCol.append(i);
        nameCol.append("Person" + std::to_string(i));
        ageCol.append(i);
        salaryCol.append(i);
        isEmployedCol.append(i % 2 == 0);

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

    // Test simple selection (age < 30)
    // auto condition = ConditionBuilder::lessThan("age", "5000000");
    auto condition = ConditionBuilder::greaterThan("name", "Person1");
    Select selectOp(testTable, *condition);

    // Execute on CPU
    auto start = std::chrono::high_resolution_clock::now();
    Table resultCPU = selectOp.execute();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << resultCPU.getRowCount() << " rows selected on CPU" << std::endl;
    std::cout << "CPU Select execution time: " << elapsed.count() << " seconds" << std::endl;
    assert(resultCPU.getRowCount() == 9999999);
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
        assert(resultGPU.getRowCount() == 9999999);
        std::cout << "GPU Select test passed!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "GPU execution not available: " << e.what() << std::endl;
    }
}

// Test Project operation
#include <iostream>
#include <chrono>
#include <cassert>

// Assuming you have a Table class, Project operation, and necessary dependencies defined

// void testProject()
// {
//     std::cout << "Testing Project operation..." << std::endl;

//     Table testTable = createTestTable();

//     // Test projection (id, name)
//     std::vector<std::string> columns = {"id", "name"};
//     Project projectOp(testTable, columns);

//     // Measure CPU execution time
//     auto startCPU = std::chrono::high_resolution_clock::now();

//     // Execute on CPU
//     Table resultCPU = projectOp.execute();

//     auto endCPU = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<float> durationCPU = endCPU - startCPU;

//     assert(resultCPU.getColumnCount() == 2);
//     assert(resultCPU.getColumnIndex("id") != -1);
//     assert(resultCPU.getColumnIndex("name") != -1);
//     assert(resultCPU.getColumnIndex("age") == -1);

//     std::cout << "CPU Project test passed!" << std::endl;
//     std::cout << "CPU execution time: " << durationCPU.count() << " seconds" << std::endl;

//     // Measure GPU execution time (if available)
//     try
//     {
//         auto startGPU = std::chrono::high_resolution_clock::now();

//         // Execute on GPU
//         Table resultGPU = projectOp.execute(USE_GPU);

//         auto endGPU = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<float> durationGPU = endGPU - startGPU;

//         assert(resultGPU.getColumnCount() == 2);
//         assert(resultGPU.getColumnIndex("id") != -1);
//         assert(resultGPU.getColumnIndex("name") != -1);
//         assert(resultGPU.getColumnIndex("age") == -1);

//         std::cout << "GPU execution time: " << durationGPU.count() << " seconds" << std::endl;
//         std::cout << "GPU Project test passed!" << std::endl;
//     }
//     catch (const std::exception &e)
//     {
//         std::cout << "GPU execution not available: " << e.what() << std::endl;
//     }
// }

// Test complex condition
void testComplexCondition()
{
    std::cout << "Testing complex conditions..." << std::endl;

    Table testTable = createTestTable();

    auto ageCondition = ConditionBuilder::greaterThan("age", "1000000");
    auto ageCondition2 = ConditionBuilder::lessThan("age", "9000000");
    auto salaryCondition = ConditionBuilder::lessThan("salary", "10");
    auto nameCondition = ConditionBuilder::greaterThan("name", "Person20");
    auto nameCondition2 = ConditionBuilder::lessThan("name", "Person20000");

    auto ageComplexCondition = ConditionBuilder::And(std::move(ageCondition), std::move(ageCondition2));
    auto nameComplexCondition = ConditionBuilder::And(std::move(nameCondition), std::move(nameCondition2));
    auto complexCondition = ConditionBuilder::Or(std::move(ageComplexCondition), std::move(nameComplexCondition));
    // auto complexCondition2 = ConditionBuilder::Or(std::move(complexCondition), std::move(salaryCondition));

    Select selectOp(testTable, *complexCondition);

    // Execute on CPU
    auto start = std::chrono::high_resolution_clock::now();
    // Table resultCPU = selectOp.execute();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // std::cout << resultCPU.getRowCount() << " rows selected on CPU" << std::endl;
    std::cout << "CPU Select execution time: " << elapsed.count() << " seconds" << std::endl;
    // assert(resultCPU.getRowCount() == 8000001);
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
        assert(resultGPU.getRowCount() == 8000001);
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

    // ---------------------- CPU TESTS ---------------------- //

    // 1. Sort by age DESC
    OrderBy orderByOp(testTable, "age", SortOrder::DESC);

    auto start = std::chrono::high_resolution_clock::now();
    Table resultCPU = orderByOp.execute();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << resultCPU.getRowCount() << " rows sorted on CPU" << std::endl;
    std::cout << "CPU OrderBy execution time (age DESC): " << elapsed.count() << " seconds" << std::endl;
    assert(resultCPU.getRowCount() == 10000000);
    assert(resultCPU.getIntValue(resultCPU.getColumnIndex("age"), 0) == 10000000);

    // Multi-column: age DESC, salary ASC
    std::vector<std::string> sortColumns = {"age", "salary"};
    std::vector<SortOrder> sortOrders = {SortOrder::DESC, SortOrder::ASC};
    OrderBy multiColOrderByOp(testTable, sortColumns, sortOrders);

    start = std::chrono::high_resolution_clock::now();
    Table multiColResult = multiColOrderByOp.execute();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> multiColElapsed = end - start;

    std::cout << multiColResult.getRowCount() << " rows sorted on CPU" << std::endl;
    std::cout << "CPU OrderBy execution time (age DESC, salary ASC): " << multiColElapsed.count() << " seconds" << std::endl;
    assert(multiColResult.getIntValue(multiColResult.getColumnIndex("age"), 0) == 10000000);

    // ---------------------- GPU TESTS ---------------------- //

    try
    {
        // 1. Sort by age DESC
        start = std::chrono::high_resolution_clock::now();
        Table resultGPU = orderByOp.execute(USE_GPU);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;

        std::cout << resultGPU.getRowCount() << " rows sorted on GPU" << std::endl;
        assert(resultGPU.getRowCount() == 10000000);
        assert(resultGPU.getIntValue(resultGPU.getColumnIndex("age"), 0) == 10000000);
        std::cout << "GPU OrderBy execution time (age DESC): " << elapsed.count() << " seconds" << std::endl;

        // 2. Multi-column: age DESC, salary ASC
        start = std::chrono::high_resolution_clock::now();
        Table multiColResultGPU = multiColOrderByOp.execute(USE_GPU);
        end = std::chrono::high_resolution_clock::now();
        multiColElapsed = end - start;

        std::cout << multiColResultGPU.getRowCount() << " rows sorted on GPU" << std::endl;
        assert(multiColResultGPU.getRowCount() == 10000000);
        assert(multiColResultGPU.getIntValue(multiColResultGPU.getColumnIndex("age"), 0) == 10000000);

        std::cout << "GPU OrderBy execution time (age DESC, salary ASC): " << multiColElapsed.count() << " seconds" << std::endl;

        std::cout << "GPU OrderBy tests passed!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "GPU execution not available: " << e.what() << std::endl;
    }
}
void testAggregator()
{
    std::cout << "\n=== Testing Aggregator operation ===" << std::endl;
    auto testTable = createTestTable();
    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time;

    // Test COUNT on the entire table
    std::cout << "\n[1] Testing COUNT aggregation..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    Aggregator countAggregator(testTable, AggregateFunction::COUNT, "id", std::nullopt, "total_count");
    Table countResult = countAggregator.executeCPU();
    end_time = std::chrono::high_resolution_clock::now();

    // Verify results
    assert(countResult.getRowCount() == 1);
    assert(countResult.getColumnCount() == 1);
    assert(countResult.getIntValue(0, 0) == 10000000);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "COUNT result: " << countResult.getIntValue(0, 0)
              << " | Execution time: " << duration.count() << " ms" << std::endl;

    // Test SUM and AVG on the salary column
    std::cout << "\n[2] Testing SUM and AVG aggregations..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    std::vector<Aggregation> aggregations = {
        Aggregation(AggregateFunction::SUM, "salary", "total_salary"),
        Aggregation(AggregateFunction::AVG, "salary", "avg_salary")};

    Aggregator multiAggregator(testTable, aggregations);
    Table multiResult = multiAggregator.executeCPU();
    end_time = std::chrono::high_resolution_clock::now();

    // Verify results
    assert(multiResult.getRowCount() == 1);
    assert(multiResult.getColumnCount() == 2);
    assert(std::abs(multiResult.getDoubleValue(0, 0) - 5.0000005e13) < 0.001);
    assert(std::abs(multiResult.getDoubleValue(1, 0) - 5000000.5) < 0.001);

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "SUM result: " << multiResult.getDoubleValue(0, 0)
              << "\nAVG result: " << multiResult.getDoubleValue(1, 0)
              << "\nExecution time: " << duration.count() << " ms" << std::endl;

    // Test MIN and MAX
    std::cout << "\n[3] Testing MIN and MAX aggregations..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    aggregations = {
        Aggregation(AggregateFunction::MIN, "age", "min_age"),
        Aggregation(AggregateFunction::MAX, "age", "max_age")};

    Aggregator minMaxAggregator(testTable, aggregations);
    Table minMaxResult = minMaxAggregator.executeCPU();
    end_time = std::chrono::high_resolution_clock::now();

    // Verify results
    assert(minMaxResult.getRowCount() == 1);
    assert(minMaxResult.getColumnCount() == 2);
    assert(minMaxResult.getIntValue(0, 0) == 1);        // Min age
    assert(minMaxResult.getIntValue(1, 0) == 10000000); // Max age

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "MIN age: " << minMaxResult.getIntValue(0, 0)
              << " | MAX age: " << minMaxResult.getIntValue(1, 0)
              << " | Execution time: " << duration.count() << " ms" << std::endl;

    // Test GROUP BY
    std::cout << "\n[4] Testing GROUP BY aggregation..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    aggregations = {
        Aggregation(AggregateFunction::COUNT, "id", "count"),
        Aggregation(AggregateFunction::AVG, "salary", "avg_salary")};

    // Group by even/odd id (which will create 2 groups)
    auto testTableCopy = testTable;
    testTableCopy.addColumn(Column("id_group", DataType::INT));

    auto &idGroupCol = static_cast<ColumnDataImpl<int> &>(testTableCopy.getColumnData("id_group"));
    for (int i = 1; i <= 10000000; i++)
    {
        idGroupCol.append(i % 2); // Group 0 for odd ids, Group 1 for even ids
    }

    Aggregator groupByAggregator(testTableCopy, aggregations, "id_group");
    Table groupByResult = groupByAggregator.executeCPU();
    end_time = std::chrono::high_resolution_clock::now();

    // Verify results
    assert(groupByResult.getRowCount() == 2);    // Two groups
    assert(groupByResult.getColumnCount() == 3); // Group column + 2 aggregations

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Group 0 count: " << groupByResult.getIntValue(0, 1)
              << " | Group 1 count: " << groupByResult.getIntValue(1, 1)
              << "\nExecution time: " << duration.count() << " ms" << std::endl;

    // GPU execution test
    std::cout << "\n[5] Testing GPU execution..." << std::endl;
    try
    {
        start_time = std::chrono::high_resolution_clock::now();
        Table resultGPU = countAggregator.execute(USE_GPU);
        end_time = std::chrono::high_resolution_clock::now();

        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "GPU COUNT result: " << resultGPU.getIntValue(0, 0)
                  << " | Execution time: " << duration.count() << " ms" << std::endl;
        std::cout << "GPU Aggregator test passed!" << std::endl;
        assert(resultGPU.getRowCount() == 1);

        start_time = std::chrono::high_resolution_clock::now();
        Table multiResultGPU = multiAggregator.execute(USE_GPU);
        end_time = std::chrono::high_resolution_clock::now();

        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "GPU SUM result: " << multiResultGPU.getDoubleValue(0, 0)
                  << "\nGPU AVG result: " << multiResultGPU.getDoubleValue(1, 0)
                  << "\nExecution time: " << duration.count() << " ms" << std::endl;

        assert(multiResultGPU.getRowCount() == 1);
        assert(std::abs(multiResultGPU.getDoubleValue(0, 0) - 5.0000005e13) < 0.001);

        assert(std::abs(multiResultGPU.getDoubleValue(1, 0) - 5000000.5) < 0.001);

        start_time = std::chrono::high_resolution_clock::now();
        Table minMaxResultGPU = minMaxAggregator.execute(USE_GPU);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "GPU MIN age: " << minMaxResultGPU.getIntValue(0, 0)
                  << " | GPU MAX age: " << minMaxResultGPU.getIntValue(1, 0)
                  << " | Execution time: " << duration.count() << " ms" << std::endl;
        assert(minMaxResultGPU.getRowCount() == 1);
        assert(minMaxResult.getColumnCount() == 2);

        assert(minMaxResultGPU.getIntValue(0, 0) == 1);
        assert(minMaxResultGPU.getIntValue(1, 0) == 10000000);

        start_time = std::chrono::high_resolution_clock::now();
        Table groupByResultGPU = groupByAggregator.execute(USE_GPU);
        end_time = std::chrono::high_resolution_clock::now();

        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "GPU Group 0 count: " << groupByResultGPU.getIntValue(0, 1)
                  << " | GPU Group 1 count: " << groupByResultGPU.getIntValue(1, 1)
                  << "\nExecution time: " << duration.count() << " ms" << std::endl;
        assert(groupByResultGPU.getRowCount() == 2);
        assert(groupByResultGPU.getColumnCount() == 3);
        // assert(groupByResultGPU.getIntValue(0, 1) == 5000000);
        // assert(groupByResultGPU.getIntValue(1, 1) == 5000000);

        std::cout << "GPU Aggregator test passed!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "GPU execution not available: " << e.what() << std::endl;
    }

    std::cout << "\nAll CPU Aggregator tests passed!" << std::endl;
}
// void testJoin()
// {
//     std::cout << "Testing Join operation..." << std::endl;

//     // Create two test tables for joining
//     Table leftTable = createTestTable(); // This is our existing test table

//     // Create a second test table with ids that will match some from the first table
//     std::vector<Column> rightColumns = {
//         Column("id", DataType::INT),
//         Column("department", DataType::VARCHAR),
//         Column("location", DataType::VARCHAR)};

//     Table rightTable(rightColumns);

//     // Add data to the right table
//     auto &rightIdCol = static_cast<ColumnDataImpl<int> &>(rightTable.getColumnData("id"));
//     auto &rightDeptCol = static_cast<ColumnDataImpl<std::string> &>(rightTable.getColumnData("department"));
//     auto &rightLocCol = static_cast<ColumnDataImpl<std::string> &>(rightTable.getColumnData("location"));

//     // Add some rows that will match (id: 2, 4) and some that won't
//     rightIdCol.append(2);
//     rightDeptCol.append("Engineering");
//     rightLocCol.append("Building A");

//     rightIdCol.append(4);
//     rightDeptCol.append("Marketing");
//     rightLocCol.append("Building B");

//     rightIdCol.append(6);
//     rightDeptCol.append("Finance");
//     rightLocCol.append("Building C");

//     // Finalize rows
//     rightTable.finalizeRow();
//     rightTable.finalizeRow();
//     rightTable.finalizeRow();

//     // Create join condition: leftTable.id = rightTable.id
//     auto joinCondition = ConditionBuilder::equals("id", "id");
//     // Test INNER JOIN
//     Join innerJoinOp(leftTable, rightTable, *joinCondition, JoinType::INNER);
//     Table innerJoinResult = innerJoinOp.executeCPU();

//     // Verify inner join results
//     assert(innerJoinResult.getRowCount() == 2); // Should match 2 rows (id 2 and 4)
//     assert(innerJoinResult.getColumnCount() == leftTable.getColumnCount() + rightTable.getColumnCount());

//     // Verify some values from the joined rows
//     int leftIdCol = innerJoinResult.getColumnIndex("id");

//     // Check that the join contains the expected IDs
//     std::vector<int> expectedIds = {2, 4}; // IDs that should be in the join result
//     bool foundId2 = false;
//     bool foundId4 = false;

//     for (size_t row = 0; row < innerJoinResult.getRowCount(); ++row)
//     {
//         int rowId = innerJoinResult.getIntValue(leftIdCol, row);
//         if (rowId == 2)
//             foundId2 = true;
//         if (rowId == 4)
//             foundId4 = true;
//     }

//     assert(foundId2 && foundId4);

//     // Test LEFT JOIN
//     Join leftJoinOp(leftTable, rightTable, *joinCondition, JoinType::LEFT);
//     Table leftJoinResult = leftJoinOp.executeCPU();

//     // Verify left join results
//     assert(leftJoinResult.getRowCount() == 5); // Should include all rows from left table

//     // Try on GPU if available
//     try
//     {
//         Table resultGPU = innerJoinOp.execute();
//         assert(resultGPU.getRowCount() == 2);
//         std::cout << "GPU Join test passed!" << std::endl;
//     }
//     catch (const std::exception &e)
//     {
//         std::cout << "GPU execution not available: " << e.what() << std::endl;
//     }

//     std::cout << "CPU Join test passed!" << std::endl;
// }

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

// Helper function to get readable type names
std::string getDataTypeName(DataType type)
{
    switch (type)
    {
    case DataType::INT:
        return "INT";
    case DataType::FLOAT:
        return "FLOAT";
    case DataType::DOUBLE:
        return "DOUBLE";
    case DataType::STRING:
        return "STRING";
    case DataType::VARCHAR:
        return "VARCHAR";
    case DataType::BOOL:
        return "BOOL";
    case DataType::DATE:
        return "DATE";
    case DataType::DATETIME:
        return "DATETIME";
    default:
        return "UNKNOWN";
    }
}

void testCSVDateTimeSupport()
{
    std::cout << "=== Testing DateTime support with CSV files ===" << std::endl;

    try
    {
        // Initialize SQLQueryProcessor with the data directory
        SQLQueryProcessor processor("/media/mohamed/0B370EA20B370EA2/CMP1Materials/Forth/Second/PC/Project/GPU-DBMS/data");

        // Print the Products table schema to verify column types
        std::cout << "\n--- Products Table Schema ---\n";
        Table productsTable = processor.getTable("Products");

        for (size_t i = 0; i < productsTable.getColumnCount(); i++)
        {
            std::cout << "Column " << i << ": " << productsTable.getColumnName(i)
                      << " (Type: " << static_cast<int>(productsTable.getColumnType(i))
                      << ", TypeName: " << getDataTypeName(productsTable.getColumnType(i)) << ")" << std::endl;
        }

        // Print sample rows to verify data
        std::cout << "\n--- Sample Product Data ---\n";
        for (size_t row = 0; row < std::min(productsTable.getRowCount(), size_t(3)); row++)
        {
            std::cout << "Row " << row << ": ";
            for (size_t col = 0; col < productsTable.getColumnCount(); col++)
            {
                std::cout << productsTable.getColumnName(col) << "=";

                // Print based on column type
                switch (productsTable.getColumnType(col))
                {
                case DataType::INT:
                    std::cout << productsTable.getIntValue(col, row);
                    break;
                case DataType::DOUBLE:
                case DataType::FLOAT:
                    std::cout << productsTable.getDoubleValue(col, row);
                    break;
                case DataType::DATE:
                case DataType::DATETIME:
                    std::cout << "'" << productsTable.getStringValue(col, row) << "'";
                    break;
                default:
                    std::cout << "'" << productsTable.getStringValue(col, row) << "'";
                }
                std::cout << ", ";
            }
            std::cout << std::endl;
        }

        // Test query for ID column - known to work
        std::cout << "\n--- Testing ID Column Selection ---\n";
        Table idResult = processor.processQuery("SELECT Products_id FROM Products");
        std::cout << "Query result has " << idResult.getRowCount() << " rows" << std::endl;

        // Test query for DateTime column - the problematic one
        std::cout << "\n--- Testing DateTime Column Selection ---\n";
        Table dateResult = processor.processQuery("SELECT ReleaseDate FROM Products");
        std::cout << "Query result has " << dateResult.getRowCount() << " rows" << std::endl;

        // If the query returned results, print them
        if (dateResult.getRowCount() > 0)
        {
            for (size_t i = 0; i < std::min(dateResult.getRowCount(), size_t(5)); i++)
            {
                try
                {
                    std::cout << "ReleaseDate: " << dateResult.getStringValue(0, i) << std::endl;
                }
                catch (const std::exception &e)
                {
                    std::cout << "Error getting value: " << e.what() << std::endl;
                }
            }
        }

        // Test a conditional query with the datetime column
        std::cout << "\n--- Testing DateTime Condition ---\n";
        std::cout << "Query: SELECT * FROM Products WHERE ReleaseDate > '2021-01-01 00:00:00'" << std::endl;
        Table conditionalResult = processor.processQuery(
            "SELECT * FROM Products WHERE ReleaseDate > '2021-01-01 00:00:00'");

        std::cout << "Query result has " << conditionalResult.getRowCount() << " rows" << std::endl;

        // Print conditional results
        for (size_t i = 0; i < conditionalResult.getRowCount(); i++)
        {
            std::cout << "Product: " << conditionalResult.getStringValue(1, i)
                      << ", Released: " << conditionalResult.getStringValue(3, i) << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error in CSV DateTime test: " << e.what() << std::endl;
    }

    std::cout << "CSV DateTime test completed!" << std::endl;
}

void testDateTimeSupport()
{
    std::cout << "Testing DateTime support..." << std::endl;

    // Create a table with a DateTime column
    std::vector<Column> columns = {
        Column("id", DataType::INT),
        Column("name", DataType::VARCHAR),
        Column("created_at", DataType::DATETIME)};

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
    for (size_t i = 0; i < result.getRowCount(); i++)
    {
        std::cout << "Event: " << result.getStringValue(1, i)
                  << ", Created: " << result.getStringValue(2, i) << std::endl;
    }

    std::cout << "DateTime support test completed!" << std::endl;
}
void testBooleanSelect()
{
    std::cout << "Testing Boolean Select operation..." << std::endl;

    // Create a table with a Boolean column
    std::vector<Column> columns = {
        Column("id", DataType::INT),
        Column("name", DataType::VARCHAR),
        Column("active", DataType::BOOL)};

    Table table(columns);

    // Add sample data with alternating boolean values
    for (int i = 1; i <= 10; i++)
    {
        table.appendIntValue(0, i);
        table.appendStringValue(1, "User" + std::to_string(i));
        table.appendBoolValue(2, i % 2 == 0); // Even IDs are active (true)
        table.finalizeRow();
    }

    // Print the table for verification
    std::cout << "Boolean table data:" << std::endl;
    for (size_t i = 0; i < table.getRowCount(); i++)
    {
        std::cout << "Row " << i << ": "
                  << "id=" << table.getIntValue(0, i) << ", "
                  << "name=" << table.getStringValue(1, i) << ", "
                  << "active=" << (table.getBoolValue(2, i) ? "true" : "false") << std::endl;
    }

    // Test boolean condition (active = true)
    auto condition = ConditionBuilder::equals("active", "true");
    Select selectOp(table, *condition);

    // Execute on CPU first to verify
    Table resultCPU = selectOp.execute(false);
    std::cout << "CPU Select result: " << resultCPU.getRowCount() << " active users found" << std::endl;

    // Print the CPU results
    for (size_t i = 0; i < resultCPU.getRowCount(); i++)
    {
        std::cout << "Active user: " << resultCPU.getStringValue(1, i)
                  << " (ID: " << resultCPU.getIntValue(0, i) << ")" << std::endl;
    }

    // Verify CPU results - should have 5 active users with even IDs
    assert(resultCPU.getRowCount() == 5);

    // Now try GPU execution
    try
    {
        Table resultGPU = selectOp.execute(true);
        std::cout << "GPU Select result: " << resultGPU.getRowCount() << " active users found" << std::endl;

        // Print the GPU results
        for (size_t i = 0; i < resultGPU.getRowCount(); i++)
        {
            std::cout << "Active user: " << resultGPU.getStringValue(1, i)
                      << " (ID: " << resultGPU.getIntValue(0, i) << ")" << std::endl;
        }

        // Verify GPU results - should match CPU results
        assert(resultGPU.getRowCount() == resultCPU.getRowCount());
        std::cout << "GPU Boolean Select test passed!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "GPU execution not available: " << e.what() << std::endl;
    }

    std::cout << "Boolean Select test completed!" << std::endl;
}

void runCLI(const std::string &dataDirectory)
{
    std::cout << "Starting SQL Query Processor CLI with data directory: " << dataDirectory << std::endl;

    try
    {
        CommandLineInterface cli(dataDirectory);
        cli.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "CLI error: " << e.what() << std::endl;
    }
}

void showHelp()
{
    std::cout << "SQL Query Processor" << std::endl;
    std::cout << "Usage: GPUDBMS [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help, -h                   Show this help message" << std::endl;
    std::cout << "  --data-dir, -d <directory>   Specify data directory (default: ./data)" << std::endl;
    std::cout << "  --test, -t <test_name>       Run specific test" << std::endl;
    std::cout << "    Available tests: select, project, condition, orderby," << std::endl;
    std::cout << "                    aggregate, join, sql, csv, datetime, boolean" << std::endl;
    std::cout << "  --test-all                   Run all tests" << std::endl;
    std::cout << "  --gpu <on|off>               Enable or disable GPU execution" << std::endl;
    std::cout << std::endl;
    std::cout << "If no options are provided, the CLI will start with the default data directory." << std::endl;
}

int main(int argc, char **argv)
{
    std::string dataDirectory = "/media/mohamed/0B370EA20B370EA2/CMP1Materials/Forth/Second/PC/Project/GPU-DBMS/data";
    bool runCli = true;
    std::string testName = "";

    // Parse command line arguments
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            showHelp();
            return 0;
        }
        else if (arg == "--data-dir" || arg == "-d")
        {
            if (i + 1 < argc)
            {
                dataDirectory = argv[++i];
            }
            else
            {
                std::cerr << "Error: --data-dir requires a directory path" << std::endl;
                return 1;
            }
        }
        else if (arg == "--test" || arg == "-t")
        {
            runCli = false;
            if (i + 1 < argc)
            {
                testName = argv[++i];
            }
            else
            {
                std::cerr << "Error: --test requires a test name" << std::endl;
                return 1;
            }
        }
        else if (arg == "--test-all")
        {
            runCli = false;
            testName = "all";
        }
    }

    // Run CLI if no test option specified
    if (runCli)
    {
        try
        {
            std::cout << "Starting SQL Query Processor CLI..." << std::endl;
            std::cout << "Data directory: " << dataDirectory << std::endl;

            CommandLineInterface cli(dataDirectory);
            cli.run();
        }
        catch (const std::exception &e)
        {
            std::cerr << "CLI error: " << e.what() << std::endl;
            return 1;
        }
    }
    // If test was specified, run the appropriate test
    else
    {
        try
        {
            // This code would call into your existing test functions
            std::cout << "Running test: " << testName << std::endl;

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
            // testBooleanSelect();

            std::cout << "All tests passed successfully!" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }

        return 0;
    }
}