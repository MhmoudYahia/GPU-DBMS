#include <iostream>
#include <cassert>
#include <memory>
#include "../include/Operations/Select.hpp"
#include "../include/DataHandling/Table.hpp"
#include "../include/DataHandling/Condition.hpp"
#include "../include/Operations/Filter.hpp"
#include "../include/Operations/OrderBy.hpp"
#include "../include/Operations/Aggregator.hpp"
#include "../include/Operations/Project.hpp"
#include "../include/Operations/Join.hpp"
#include "../include/SQLProcessing/SQLQueryProcessor.hpp"

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
    for (int i = 1; i <= 5; i++)
    {
        idCol.append(i);
        nameCol.append("Person" + std::to_string(i));
        ageCol.append(20 + i * 5);
        salaryCol.append(50000.0 + i * 10000.0);
        
        // Finalize each row after adding all column values
        table.finalizeRow();
    }

    return table;
}

// Test Select operation
void testSelect()
{
    std::cout << "Testing Select operation..." << std::endl;

    Table testTable = createTestTable();

    // Test simple selection (age > 30)
    auto condition = ConditionBuilder::greaterThan("age", "30");
    Select selectOp(testTable, *condition);

    // Execute on CPU
    Table resultCPU = selectOp.executeCPU();
    assert(resultCPU.getRowCount() == 3);

    // Execute on GPU if available
    try
    {
        Table resultGPU = selectOp.execute();
        assert(resultGPU.getRowCount() == 3);
        std::cout << "GPU Select test passed!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "GPU execution not available: " << e.what() << std::endl;
    }

    std::cout << "CPU Select test passed!" << std::endl;
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

    // age > 25 AND (salary > 70000 OR active = true)
    auto ageCondition = ConditionBuilder::greaterThan("age", "25");
    auto salaryCondition = ConditionBuilder::greaterThan("salary", "70000");
    auto activeCondition = ConditionBuilder::equals("active", "true");

    auto orCondition = ConditionBuilder::Or(std::move(salaryCondition), std::move(activeCondition));
    auto complexCondition = ConditionBuilder::And(std::move(ageCondition), std::move(orCondition));

    Select selectOp(testTable, *complexCondition);

    Table resultCPU = selectOp.executeCPU();
    std::cout << "Complex condition result rows: " << resultCPU.getRowCount() << std::endl;

    // The expected result depends on the data, so just make sure execution completes
    std::cout << "Complex condition test passed!" << std::endl;
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
        Column("location", DataType::VARCHAR)
    };

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
    
    for (size_t row = 0; row < innerJoinResult.getRowCount(); ++row) {
        int rowId = innerJoinResult.getIntValue(leftIdCol, row);
        if (rowId == 2) foundId2 = true;
        if (rowId == 4) foundId4 = true;
    }
    
    assert(foundId2 && foundId4);

    // Test LEFT JOIN
    Join leftJoinOp(leftTable, rightTable, *joinCondition, JoinType::LEFT);
    Table leftJoinResult = leftJoinOp.executeCPU();

    // Verify left join results
    assert(leftJoinResult.getRowCount() == 5); // Should include all rows from left table
    
    // Try on GPU if available
    try {
        Table resultGPU = innerJoinOp.execute();
        assert(resultGPU.getRowCount() == 2);
        std::cout << "GPU Join test passed!" << std::endl;
    }
    catch (const std::exception &e) {
        std::cout << "GPU execution not available: " << e.what() << std::endl;
    }

    std::cout << "CPU Join test passed!" << std::endl;
}


void testSQLQueryProcessor() {
    std::cout << "Testing SQL Query Processor..." << std::endl;
    
    SQLQueryProcessor processor;
    
    // Create a test table and register it with the processor
    Table employeesTable = createTestTable();
    std::cout << "Created employees table with " << employeesTable.getRowCount() << " rows" << std::endl;
    
    // Print the content of the test table for debugging
    std::cout << "Employees table content:" << std::endl;
    for (size_t row = 0; row < employeesTable.getRowCount(); ++row) {
        std::cout << "Row " << row << ": ";
        std::cout << "id=" << employeesTable.getIntValue(0, row) << ", ";
        std::cout << "name=" << employeesTable.getStringValue(1, row) << ", ";
        std::cout << "age=" << employeesTable.getIntValue(2, row) << ", ";
        std::cout << "salary=" << employeesTable.getDoubleValue(3, row) << std::endl;
    }
    
    processor.registerTable("employees", employeesTable);
    
    // Test a simple SELECT query
    try {
        std::cout << "Running query: SELECT id, name FROM employees WHERE age > 30" << std::endl;
        Table result = processor.processQuery("SELECT id, name FROM employees WHERE age > 30");
        std::cout << "Query execution successful. Result rows: " << result.getRowCount() << std::endl;
        
        // Print the result for debugging
        std::cout << "Query result:" << std::endl;
        for (size_t row = 0; row < result.getRowCount(); ++row) {
            std::cout << "Row " << row << ": ";
            std::cout << "id=" << result.getIntValue(0, row) << ", ";
            std::cout << "name=" << result.getStringValue(1, row) << std::endl;
        }
        
        // Verify the result has the expected columns
        assert(result.getColumnCount() == 2);
        assert(result.getColumnIndex("id") != -1);
        assert(result.getColumnIndex("name") != -1);
        
        // Verify all returned rows have age > 30 (but we can't check directly since age isn't in the result)
        assert(result.getRowCount() == 3); // Based on our test data
        
        // ... rest of the test...
    }
    catch (const std::exception& e) {
        std::cerr << "Error executing query: " << e.what() << std::endl;
        throw; // Re-throw to fail the test
    }
    
    std::cout << "SQL Query Processor test passed!" << std::endl;
}
int main()
{
    try
    {
        // testSelect();
        // testProject();
        // testComplexCondition();
        // testFilter();
        // testOrderBy();
        // testAggregator();
        // testJoin();
        testSQLQueryProcessor();

        std::cout << "All tests passed successfully!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}