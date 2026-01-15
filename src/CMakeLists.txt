add_library(mokert "")
add_subdirectory(native/${MOKERT_PLATFORM})
target_include_directories(mokert PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(mokert PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
