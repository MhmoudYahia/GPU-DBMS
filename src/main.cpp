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
#include "../include/Operations/Join.hpp"
#include "../include/SQLProcessing/SQLQueryProcessor.hpp"
#include "../include/CLI/CommandLineInterface.hpp"
#define USE_GPU 1

using namespace GPUDBMS;

// Helper function to create a test table
Table createTestTable()
{
    std::vector<Column> columns = {
        Column("Product_id", DataType::INT),
        Column("ProductName", DataType::VARCHAR),
        Column("Price", DataType::DOUBLE),
        Column("ReleaseDate", DataType::DATETIME),
        Column("Trending", DataType::BOOL),
    };

    Table table(columns);

    auto &idCol = static_cast<ColumnDataImpl<int> &>(table.getColumnData("Product_id"));
    auto &nameCol = static_cast<ColumnDataImpl<std::string> &>(table.getColumnData("ProductName"));
    auto &priceCol = static_cast<ColumnDataImpl<double> &>(table.getColumnData("Price"));
    auto &dateCol = static_cast<ColumnDataImpl<std::string> &>(table.getColumnData("ReleaseDate"));
    auto &trendingCol = static_cast<ColumnDataImpl<bool> &>(table.getColumnData("Trending"));

    // Base dataset (26 entries)
    std::vector<std::tuple<int, std::string, double, std::string, bool>> baseData = {
        {101, "Widget A", 20.99, "2020-05-07 14:30:00", true},
        {102, "Gadget B", 99.99, "2021-04-22 16:10:00", false},
        {103, "Device C", 15.5, "2021-09-30 11:20:00", true},
        {104, "Equipment D", 45.75, "2022-05-15 09:45:00", false},
        {105, "Appliance E", 30.0, "2019-01-12 08:00:00", true},
        {106, "Tool F", 75.49, "2022-07-18 13:25:00", true},
        {107, "Component G", 12.99, "2020-11-03 10:15:00", true},
        {108, "Accessory H", 8.75, "2021-12-05 17:40:00", false},
        {109, "Instrument J", 149.99, "2022-02-28 14:50:00", true},
        {110, "System K", 199.99, "2019-06-14 09:30:00", false},
        {111, "Module L", 55.25, "2020-08-22 11:45:00", false},
        {112, "Kit M", 89.95, "2021-10-17 15:20:00", false},
        {113, "Unit N", 32.5, "2022-01-09 12:10:00", true},
        {114, "Bundle P", 125.0, "2019-09-29 10:05:00", true},
        {115, "Package Q", 67.8, "2020-03-11 16:35:00", false},
        {116, "Set R", 42.99, "2021-07-25 13:55:00", true},
        {117, "Assembly S", 95.45, "2022-04-03 09:15:00", false},
        {118, "Collection T", 110.25, "2019-11-18 14:40:00", true},
        {119, "Solution U", 135.5, "2020-12-07 10:50:00", false},
        {120, "Platform V", 175.0, "2021-05-31 16:25:00", true},
        {121, "Widget Pro", 24.99, "2020-06-15 13:30:00", true},
        {122, "Gadget XL", 119.99, "2021-05-12 14:15:00", false},
        {123, "Device Mini", 12.5, "2021-08-20 10:20:00", false},
        {124, "Equipment Plus", 55.75, "2022-04-25 08:45:00", true},
        {125, "Appliance Max", 35.0, "2019-02-22 09:10:00", true},
        {126, "Tool Set", 95.49, "2022-08-28 12:25:00", false},
    };

    int idOffset = 0;
    for (int i = 0; i < 1000000; ++i)
    {
        const auto &[idBase, nameBase, priceBase, dateBase, trendingBase] = baseData[i % baseData.size()];

        int newId = idBase + idOffset;
        std::string newName = nameBase + " #" + std::to_string(i + 1);
        double newPrice = priceBase + (i % 5) * 1.1; // vary the price a bit
        std::string newDate = dateBase;

        bool newTrending = trendingBase;

        idCol.append(newId);
        nameCol.append(newName);
        priceCol.append(newPrice);
        dateCol.append(newDate);
        trendingCol.append(newTrending);

        table.finalizeRow();

        if ((i + 1) % baseData.size() == 0)
            idOffset += 100; // increase ID range to avoid repetition
    }

    return table;
}

Table createTestTable2()
{
    std::vector<Column> columns = {
        Column("id", DataType::INT),
        Column("age", DataType::INT),
        Column("salary", DataType::DOUBLE),
        Column("name", DataType::VARCHAR),
        Column("retired", DataType::BOOL),
        Column("releaseDate", DataType::DATETIME),
    };

    Table table(columns);

    auto &idCol = static_cast<ColumnDataImpl<int> &>(table.getColumnData("id"));
    auto &ageCol = static_cast<ColumnDataImpl<int> &>(table.getColumnData("age"));
    auto &salaryCol = static_cast<ColumnDataImpl<double> &>(table.getColumnData("salary"));
    auto &nameCol = static_cast<ColumnDataImpl<std::string> &>(table.getColumnData("name"));
    auto &retiredCol = static_cast<ColumnDataImpl<bool> &>(table.getColumnData("retired"));
    auto &dateCol = static_cast<ColumnDataImpl<std::string> &>(table.getColumnData("releaseDate"));

    int idOffset = 0;
    for (int i = 0; i < 10000000; ++i)
    {

        int newId = i + idOffset;
        int newAge = 20 + (i % 50);                   // age between 20 and 69
        double newSalary = 30000 + (i % 100) * 100.0; // salary between 30,000 and 40,000
        std::string newName = "Person" + std::to_string(i + 1);
        bool newRetired = (i % 2 == 0);              // alternate between true and false
        // Generate a realistic date range (from 2020-01-01 to 2023-12-31)
        int year = 2020 + (i % 4);  // Years from 2020 to 2023 
        int month = 1 + (i % 12);   // Months from 1 to 12
        int day = 1 + (i % 28);     // Days from 1 to 28 (avoiding month end issues)
        int hour = i % 24;          // Hours from 0 to 23
        int minute = i % 60;        // Minutes from 0 to 59
        int second = i % 60;        // Seconds from 0 to 59
        
        char dateBuffer[50];
        snprintf(dateBuffer, sizeof(dateBuffer), "%04d-%02d-%02d %02d:%02d:%02d", 
             year, month, day, hour, minute, second);
        std::string newDate = dateBuffer;

        idCol.append(newId);
        ageCol.append(newAge);
        salaryCol.append(newSalary);
        nameCol.append(newName);
        retiredCol.append(newRetired);
        dateCol.append(newDate);

        // table.finalizeRow();
        if ((i + 1) % 100000 == 0)
            idOffset += 100; // increase ID range to avoid repetition
    }

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

void testProject()
{
    std::cout << "Testing Project operation..." << std::endl;

    Table testTable = createTestTable2();

    std::vector<std::string> columns = {"age", "salary", "retired"};
    Project projectOp(testTable, columns);

    // Measure CPU execution time
    auto startCPU = std::chrono::high_resolution_clock::now();

    // Execute on CPU
    Table resultCPU = projectOp.execute();

    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> durationCPU = endCPU - startCPU;

    assert(resultCPU.getColumnCount() == 3);
    assert(resultCPU.getRowCount() == testTable.getRowCount());
    assert(resultCPU.getColumnIndex("age") != -1);
    assert(resultCPU.getColumnIndex("salary") != -1);
    assert(resultCPU.getColumnIndex("retired") != -1);
    assert(resultCPU.getColumnIndex("id") == -1);

    std::cout << "CPU Project test passed!" << std::endl;
    std::cout << "CPU execution time: " << durationCPU.count() << " seconds" << std::endl;

    // Measure GPU execution time (if available)
    try
    {
        auto startGPU = std::chrono::high_resolution_clock::now();

        // Execute on GPU
        Table resultGPU = projectOp.execute(USE_GPU);

        auto endGPU = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> durationGPU = endGPU - startGPU;

        assert(resultGPU.getColumnCount() == 3);
        assert(resultGPU.getRowCount() == testTable.getRowCount());
        assert(resultGPU.getColumnIndex("age") != -1);
        assert(resultGPU.getColumnIndex("salary") != -1);
        assert(resultGPU.getColumnIndex("retired") != -1);
        assert(resultGPU.getColumnIndex("id") == -1);


        std::cout << "GPU execution time: " << durationGPU.count() << " seconds" << std::endl;
        std::cout << "GPU Project test passed!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "GPU execution not available: " << e.what() << std::endl;
    }
}

// Test complex condition
void testComplexCondition()
{
    std::cout << "Testing complex conditions..." << std::endl;

    Table testTable = createTestTable2();

    auto priceCondition = ConditionBuilder::greaterThan("age", "20");
    auto priceCondition2 = ConditionBuilder::lessThan("age", "50");
    // auto salaryCondition = ConditionBuilder::lessThan("salary", "35000");
    auto nameCondition = ConditionBuilder::greaterThan("name", "Person20");
    auto nameCondition2 = ConditionBuilder::lessThan("name", "Person20000");

    auto priceComplexCondition = ConditionBuilder::And(std::move(priceCondition), std::move(priceCondition2));
    // auto nameComplexCondition = ConditionBuilder::And(std::move(nameCondition), std::move(nameCondition2));
    // auto complexCondition = ConditionBuilder::Or(std::move(ageComplexCondition), std::move(nameComplexCondition));

    Select selectOp(testTable, *priceComplexCondition);

    std::cout << "Executing complex condition on cpu..." << std::endl;
    // Execute on CPU
    auto start = std::chrono::high_resolution_clock::now();
    Table resultCPU = selectOp.execute();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << resultCPU.getRowCount() << " rows selected on CPU" << std::endl;
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
        // assert(resultGPU.getRowCount() == 8000001);
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

    Table testTable = createTestTable2();

    // ---------------------- CPU TESTS ---------------------- //

    // 1. Sort by age DESC
    OrderBy orderByOp(testTable, "age", SortOrder::DESC);

    auto start = std::chrono::high_resolution_clock::now();
    Table resultCPU = orderByOp.execute();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << resultCPU.getRowCount() << " rows sorted on CPU" << std::endl;
    std::cout << "CPU OrderBy execution time (age DESC): " << elapsed.count() << " seconds" << std::endl;
    assert(resultCPU.getRowCount() == testTable.getRowCount());

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

    // ---------------------- GPU TESTS ---------------------- //

    try
    {
        // 1. Sort by age DESC
        start = std::chrono::high_resolution_clock::now();
        Table resultGPU = orderByOp.execute(USE_GPU);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;

        std::cout << resultGPU.getRowCount() << " rows sorted on GPU" << std::endl;
        assert(resultGPU.getRowCount() == testTable.getRowCount());
        std::cout << "GPU OrderBy execution time (age DESC): " << elapsed.count() << " seconds" << std::endl;

        // 2. Multi-column: age DESC, salary ASC
        start = std::chrono::high_resolution_clock::now();
        Table multiColResultGPU = multiColOrderByOp.execute(USE_GPU);
        end = std::chrono::high_resolution_clock::now();
        multiColElapsed = end - start;

        std::cout << multiColResultGPU.getRowCount() << " rows sorted on GPU" << std::endl;
        assert(multiColResultGPU.getRowCount() == testTable.getRowCount());

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
    auto testTable = createTestTable2();
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

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "MIN age: " << minMaxResult.getIntValue(0, 0)
              << " | MAX age: " << minMaxResult.getIntValue(1, 0)
              << " | Execution time: " << duration.count() << " ms" << std::endl;

    // // Test GROUP BY
    // std::cout << "\n[4] Testing GROUP BY aggregation..." << std::endl;
    // start_time = std::chrono::high_resolution_clock::now();
    // aggregations = {
    //     Aggregation(AggregateFunction::COUNT, "id", "count"),
    //     Aggregation(AggregateFunction::AVG, "salary", "avg_salary")};

    // // Group by even/odd id (which will create 2 groups)
    // auto testTableCopy = testTable;
    // testTableCopy.addColumn(Column("id_group", DataType::INT));

    // auto &idGroupCol = static_cast<ColumnDataImpl<int> &>(testTableCopy.getColumnData("id_group"));
    // for (int i = 1; i <= 10000000; i++)
    // {
    //     idGroupCol.append(i % 2); // Group 0 for odd ids, Group 1 for even ids
    // }

    // Aggregator groupByAggregator(testTableCopy, aggregations, "id_group");
    // Table groupByResult = groupByAggregator.executeCPU();
    // end_time = std::chrono::high_resolution_clock::now();

    // // Verify results
    // assert(groupByResult.getRowCount() == 2);    // Two groups
    // assert(groupByResult.getColumnCount() == 3); // Group column + 2 aggregations

    // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    // std::cout << "Group 0 count: " << groupByResult.getIntValue(0, 1)
    //           << " | Group 1 count: " << groupByResult.getIntValue(1, 1)
    //           << "\nExecution time: " << duration.count() << " ms" << std::endl;

    // std::vector<Aggregation> dateTime = {
    //     Aggregation(AggregateFunction::COUNT, "releaseDate", "count"),
    //     Aggregation(AggregateFunction::MAX, "releaseDate", "avg_date"),
    //     Aggregation(AggregateFunction::MIN, "releaseDate", "min_date")};

    // Aggregator dateTimeAggregator(testTable, dateTime);

    // start_time = std::chrono::high_resolution_clock::now();
    // Table dateTimeResult = dateTimeAggregator.executeCPU();
    // end_time = std::chrono::high_resolution_clock::now();

    // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // std::cout << "DateTime COUNT result: " << dateTimeResult.getIntValue(0, 0)
    //           << " | DateTime MAX result: " << dateTimeResult.getStringValue(1, 0)
    //           << " | DateTime MIN result: " << dateTimeResult.getStringValue(2, 0)
    //           << "\nExecution time: " << duration.count() << " ms" << std::endl;
    // assert(dateTimeResult.getRowCount() == 1);
    // assert(dateTimeResult.getColumnCount() == 3);

    // GPU execution test
    std::cout << "\n[5] Testing GPU execution..." << std::endl;
    try
    {
        // start_time = std::chrono::high_resolution_clock::now();
        // Table resultGPU = countAggregator.execute(USE_GPU);
        // end_time = std::chrono::high_resolution_clock::now();

        // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // std::cout << "GPU COUNT result: " << resultGPU.getIntValue(0, 0)
        //           << " | Execution time: " << duration.count() << " ms" << std::endl;
        // std::cout << "GPU Aggregator test passed!" << std::endl;
        // assert(resultGPU.getRowCount() == 1);

        start_time = std::chrono::high_resolution_clock::now();
        Table multiResultGPU = multiAggregator.execute(USE_GPU);
        end_time = std::chrono::high_resolution_clock::now();

        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "GPU SUM result: " << multiResultGPU.getDoubleValue(0, 0)
                  << "\nGPU AVG result: " << multiResultGPU.getDoubleValue(1, 0)
                  << "\nExecution time: " << duration.count() << " ms" << std::endl;

        assert(multiResultGPU.getRowCount() == 1);


        start_time = std::chrono::high_resolution_clock::now();
        Table minMaxResultGPU = minMaxAggregator.execute(USE_GPU);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "GPU MIN age: " << minMaxResultGPU.getIntValue(0, 0)
                  << " | GPU MAX age: " << minMaxResultGPU.getIntValue(1, 0)
                  << " | Execution time: " << duration.count() << " ms" << std::endl;
        assert(minMaxResultGPU.getRowCount() == 1);
        assert(minMaxResult.getColumnCount() == 2);


        // start_time = std::chrono::high_resolution_clock::now();
        // Table dateTimeResultGPU = dateTimeAggregator.execute(USE_GPU);
        // end_time = std::chrono::high_resolution_clock::now();
        // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // std::cout 
        //           << " | GPU DateTime MAX result: " << dateTimeResultGPU.getStringValue(1, 0)
        //           << " | GPU DateTime MIN result: " << dateTimeResultGPU.getStringValue(2, 0)
        //           << "\nExecution time: " << duration.count() << " ms" << std::endl;

        // assert(dateTimeResultGPU.getRowCount() == 1);
        // assert(dateTimeResultGPU.getColumnCount() == 3);

        // start_time = std::chrono::high_resolution_clock::now();
        // Table groupByResultGPU = groupByAggregator.execute(USE_GPU);
        // end_time = std::chrono::high_resolution_clock::now();

        // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        // std::cout << "GPU Group 0 count: " << groupByResultGPU.getIntValue(0, 1)
        //           << " | GPU Group 1 count: " << groupByResultGPU.getIntValue(1, 1)
        //           << "\nExecution time: " << duration.count() << " ms" << std::endl;
        // assert(groupByResultGPU.getRowCount() == 2);
        // assert(groupByResultGPU.getColumnCount() == 3);

        std::cout << "GPU Aggregator test passed!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "GPU execution not available: " << e.what() << std::endl;
    }

    std::cout << "\nAll CPU Aggregator tests passed!" << std::endl;
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

void testProductOrderJoin()
{

    std::cout << "=== Testing SalesOrders-Products JOIN ===" << std::endl;

    try
    {
        // Initialize SQLQueryProcessor with the data directory
        SQLQueryProcessor processor("/media/mohamed/0B370EA20B370EA2/CMP1Materials/Forth/Second/PC/Project/GPU-DBMS/data");

        // Print table information to verify tables are loaded correctly
        std::cout << "\n--- Products Table Info ---" << std::endl;
        Table productsTable = processor.getTable("Products");
        std::cout << "Products table loaded with " << productsTable.getRowCount() << " rows and "
                  << productsTable.getColumnCount() << " columns" << std::endl;

        std::cout << "\n--- SalesOrders Table Info ---" << std::endl;
        Table salesOrdersTable = processor.getTable("SalesOrders");
        std::cout << "SalesOrders table loaded with " << salesOrdersTable.getRowCount() << " rows and "
                  << salesOrdersTable.getColumnCount() << " columns" << std::endl;

        std::cout << "\nExecuting query: SELECT o.Orders_id, o.CustomerName, p.ProductName, p.Price, o.TotalAmount "
                  << "FROM SalesOrders o JOIN Products p ON o.Products_id = p.Products_id" << std::endl;

        // Execute the query and measure performance
        auto start = std::chrono::high_resolution_clock::now();

        Table result = processor.processQuery(
            "SELECT o.Orders_id, o.CustomerName, p.ProductName, p.Price, o.TotalAmount "
            "FROM SalesOrders o JOIN Products p ON o.Products_id = p.Products_id;");

        result.printTableInfo(); // Print table info for debugging

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Print the results
        std::cout << "\nJOIN Result (" << result.getRowCount() << " rows) | Execution time: "
                  << duration.count() << " ms" << std::endl;

        // Display column names
        std::cout << "\nColumns: ";
        for (size_t i = 0; i < result.getColumnCount(); i++)
        {
            std::cout << result.getColumnName(i);
            if (i < result.getColumnCount() - 1)
                std::cout << ", ";
        }
        std::cout << std::endl;

        // Print the first few result rows
        for (size_t i = 0; i < result.getRowCount(); i++)
        {
            std::cout << "Row " << i << ": ";
            std::cout << "Orders_id=" << result.getDoubleValue(0, i)
                      << ", CustomerName=" << result.getStringValue(1, i)
                      << ", ProductName=" << result.getStringValue(7, i)
                      << ", Price=" << result.getDoubleValue(8, i)
                      << ", TotalAmount=" << result.getDoubleValue(2, i) << std::endl;
        }

        std::cout << "\nSalesOrders-Products JOIN test completed!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error in Product-Order JOIN test: " << e.what() << std::endl;
    }
}

void testDirectJoin()
{
    std::cout << "=== Testing Direct Join Operation ===" << std::endl;

    try
    {
        // Load tables directly from CSV files
        std::string dataDir = "/media/mohamed/0B370EA20B370EA2/CMP1Materials/Forth/Second/PC/Project/GPU-DBMS/data";
        CSVProcessor csvProcessor;

        // Load SalesOrders table
        std::cout << "\nLoading SalesOrders table from CSV..." << std::endl;
        Table salesOrdersTable = csvProcessor.readCSV(dataDir + "/input_csvs/SalesOrders.csv");
        std::cout << "SalesOrders table loaded with " << salesOrdersTable.getRowCount()
                  << " rows and " << salesOrdersTable.getColumnCount() << " columns" << std::endl;

        // Load Products table
        std::cout << "\nLoading Products table from CSV..." << std::endl;
        Table productsTable = csvProcessor.readCSV(dataDir + "/input_csvs/Products.csv");
        std::cout << "Products table loaded with " << productsTable.getRowCount()
                  << " rows and " << productsTable.getColumnCount() << " columns" << std::endl;

        // Print sample data from both tables
        std::cout << "\n--- SalesOrders Sample ---" << std::endl;
        for (size_t i = 0; i < std::min(salesOrdersTable.getRowCount(), size_t(3)); i++)
        {
            std::cout << "Orders_id: " << salesOrdersTable.getDoubleValue(0, i)
                      << ", CustomerName: " << salesOrdersTable.getStringValue(1, i)
                      << ", Products_id: " << salesOrdersTable.getDoubleValue(4, i) << std::endl;
        }

        std::cout << "\n--- Products Sample ---" << std::endl;
        for (size_t i = 0; i < std::min(productsTable.getRowCount(), size_t(3)); i++)
        {
            std::cout << "Products_id: " << productsTable.getDoubleValue(0, i)
                      << ", ProductName: " << productsTable.getStringValue(1, i)
                      << ", Price: " << productsTable.getDoubleValue(2, i) << std::endl;
        }

        // Create join condition: SalesOrders.Products_id = Products.Products_id
        auto condition = ConditionBuilder::columnEquals("Products_id", "Products_id");

        // Create and execute the join
        std::cout << "\nExecuting JOIN operation directly..." << std::endl;
        Join joinOp(salesOrdersTable, productsTable, *condition);

        auto start = std::chrono::high_resolution_clock::now();
        Table joinResult = joinOp.execute(false); // Use CPU implementation
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "JOIN completed with " << joinResult.getRowCount()
                  << " rows in " << duration.count() << " ms" << std::endl;

        // Print column names of the result
        std::cout << "\n--- Join Result Columns ---" << std::endl;
        for (size_t i = 0; i < joinResult.getColumnCount(); i++)
        {
            std::cout << i << ": " << joinResult.getColumnName(i)
                      << " (" << getDataTypeName(joinResult.getColumnType(i)) << ")" << std::endl;
        }

        // Print a few rows from the join result
        std::cout << "\n--- Join Result Sample ---" << std::endl;
        size_t resultColCount = joinResult.getColumnCount();
        size_t rightProductNameIdx = 0;
        size_t rightPriceIdx = 0;
        size_t rightProductIdIdx = 0;

        // Find right table column indices (they might have been renamed)
        for (size_t i = 0; i < resultColCount; i++)
        {
            std::string colName = joinResult.getColumnName(i);
            if (colName == "ProductName" && i >= salesOrdersTable.getColumnCount())
            {
                rightProductNameIdx = i;
            }
            else if (colName == "Price" && i >= salesOrdersTable.getColumnCount())
            {
                rightPriceIdx = i;
            }
            else if (colName == "Products_id" && i >= salesOrdersTable.getColumnCount())
            {
                rightProductIdIdx = i;
            }
        }

        for (size_t i = 0; i < joinResult.getRowCount(); i++)
        // for (size_t i = 0; i < 5; i++)

        {
            std::cout << "Row " << i << ": "
                      << "Orders_id=" << joinResult.getDoubleValue(0, i)
                      << ", CustomerName=" << joinResult.getStringValue(1, i)
                      << ", Products_id=" << joinResult.getDoubleValue(4, i);

            // Add right table columns if found
            if (rightProductNameIdx > 0)
            {
                std::cout << ", ProductName=" << joinResult.getStringValue(rightProductNameIdx, i);
            }
            if (rightPriceIdx > 0)
            {
                std::cout << ", Price=" << joinResult.getDoubleValue(rightPriceIdx, i);
            }

            std::cout << std::endl;
        }

        // Verify expected results
        std::cout << "\n--- Validation ---" << std::endl;
        // We should get one row for each valid Products_id match
        std::unordered_map<int, bool> matchedProducts;
        for (size_t i = 0; i < joinResult.getRowCount(); i++)
        {
            int leftProductId = joinResult.getDoubleValue(4, i); // SalesOrders.Products_id

            if (rightProductIdIdx > 0)
            {
                int rightProductId = joinResult.getDoubleValue(rightProductIdIdx, i);

                // Verify join condition is satisfied
                if (leftProductId != rightProductId)
                {
                    std::cout << "ERROR: Join condition violated! " << leftProductId
                              << " != " << rightProductId << std::endl;
                }
            }

            matchedProducts[leftProductId] = true;
        }

        std::cout << "Found " << matchedProducts.size() << " unique product IDs in join result" << std::endl;
        std::cout << "Direct Join test completed successfully!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error in Direct Join test: " << e.what() << std::endl;
    }
}

int main(int argc, char **argv)
{
    std::string dataDirectory = "/media/mohamed/0B370EA20B370EA2/CMP1Materials/Forth/Second/PC/Project/GPU-DBMS/data";
    bool runCli = false;
    // std::string dataDirectory = "/mnt/g/MyRepos/SQLQueryProcessor/data";
    // bool runCli = false;
    // std::string testName = "";

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
                // testName = argv[++i];
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
            // testName = "all";
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
            // std::cout << "Running test: " << testName << std::endl;

            // testSelect();
            // testProject();
            // testProject();
            // testComplexCondition();
            // testFilter();
            // testOrderBy();
            testAggregator();
            // testJoin();
            // testEmployeeOrderJoin();
            // testProductOrderJoin();
            // testDirectJoin();
            // testSQLQueryProcessor();
            // testCSVLoading();
            // testDateTimeSupport();
            // testBooleanSelect();
            // testCSVDateTimeSupport();

            std::cout
                << "All tests passed successfully!" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }

        return 0;
    }
}