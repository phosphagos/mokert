cmake_minimum_required(VERSION 3.22.0)
project(mokert VERSION 0.1.0)

set(MOKERT_PLATFORM cuda CACHE STRING "target platform.")
option(MOKERT_ENABLE_TESTS "enable building unit tests" OFF)

include(cmake/cpp.cmake)
include(cmake/${MOKERT_PLATFORM}.cmake)

add_subdirectory(src)
if(MOKERT_ENABLE_TESTS)
    add_subdirectory(test)
endif()
