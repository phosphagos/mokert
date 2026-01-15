target_sources(
    mokert

    PRIVATE
    runtime.cpp
    memory.cu
    algorithm.cu
    timer.cpp
)

target_compile_definitions(mokert PUBLIC -DMOKE_PLATFORM_CUDA)
target_link_libraries(mokert PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::curand)
