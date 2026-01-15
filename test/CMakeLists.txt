include(CTest)
enable_testing()
find_package(GTest CONFIG REQUIRED)

macro(mokert_add_test test_name)
    add_executable(${test_name} ${ARGN})
    target_link_libraries(${test_name} PRIVATE mokert)
    target_link_libraries(${test_name} PRIVATE GTest::gtest GTest::gtest_main)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(AllTestInMain ${test_name})
endmacro()

mokert_add_test(
    test_mokert_native
    native/test_memory.cpp
    native/test_algorithm.cpp
    native/test_timer.cpp
)

mokert_add_test(
    test_mokert_runtime
    runtime/test_vector.cpp
    runtime/test_functional.cpp
    runtime/test_profiler.cpp
)