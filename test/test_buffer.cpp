#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include "base/buffer.h"

TEST(test_allocate, allocate_cpu) {
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    hpinfer::Buffer buffer(1280, alloc_cpu);
    ASSERT_NE(buffer.ptr(), nullptr);
}

TEST(test_allocate, allocate_cuda) {
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
    hpinfer::Buffer buffer(1280, alloc_cu);
    ASSERT_NE(buffer.ptr(), nullptr);
}

