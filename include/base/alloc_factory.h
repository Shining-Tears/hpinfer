#pragma once

#include "base/alloc.h"

class CPUDeviceAllocatorFactory {
    static std::shared_ptr<CPUDeviceAllocator> instance;
    static std::once_flag initInstanceFlag;

    CPUDeviceAllocatorFactory() = default;

public:
    CPUDeviceAllocatorFactory(const CPUDeviceAllocator &) = delete;
    CPUDeviceAllocatorFactory &operator = (const CPUDeviceAllocator &) = delete;

    static std::shared_ptr<CPUDeviceAllocator> get_instance();
};

class CUDADeviceAllocatorFactory {
    static std::shared_ptr<CUDADeviceAllocator> instance;
    static std::once_flag initInstanceFlag;

    CUDADeviceAllocatorFactory() = default;

public:
    CUDADeviceAllocatorFactory(const CUDADeviceAllocator &) = delete;
    CUDADeviceAllocatorFactory &operator = (const CUDADeviceAllocator &) = delete;

    static std::shared_ptr<CUDADeviceAllocator> get_instance();
};

