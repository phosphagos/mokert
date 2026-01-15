cmake_minimum_required(VERSION 3.22.0)
project(mokert VERSION 0.1.0)
option(MOKE_ENABLE_TESTS "enable building unit tests" OFF)

include(cmake/cpp.cmake)
include(cmake/${MOKE_PLATFORM}.cmake)

add_subdirectory(src)
if(MOKE_ENABLE_TESTS)
    add_subdirectory(test)
endif()
