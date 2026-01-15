target_sources(
    mokert

    PRIVATE
    runtime.cpp
    memory.hip
    algorithm.hip
    timer.cpp
)

target_compile_definitions(mokert PUBLIC -DMOKERT_PLATFORM_HIP)
target_link_libraries(mokert PUBLIC hip::host)
