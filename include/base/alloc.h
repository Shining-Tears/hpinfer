#pragma once

#include <map>
#include <memory>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>

enum class DeviceType {
    kDeviceUnkown = 0,
    kDeviceCPU = 1,
    kDeviceCUDA = 2,
};

enum class MemcpyKind {
    kMemcpyCPU2CPU = 0,
    kMemcpyCPU2CUDA = 1,
    kMemcpyCUDA2CPU = 0,
    kMemcpyCUDA2CUDA = 1,
};

class DeviceAllocator {
    DeviceType device_type_;
public:
    explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {};

    virtual void* allocate(size_t byte_size) const = 0;

    virtual void memcpy(void* dst, const void* src, size_t byte_size, MemcpyKind memcpy_kind, 
                        void* stream = nullptr, bool need_sync = false) const;

    virtual void release(void* ptr) const = 0;

    virtual DeviceType device_type() const { return device_type_; };
};

class CPUDeviceAllocator: public DeviceAllocator {
public:
    explicit CPUDeviceAllocator();

    void* allocate(size_t byte_size) const override;

    void release(void* ptr) const override;
};

class CUDADeviceAllocator: public DeviceAllocator {
public:
    explicit CUDADeviceAllocator();

    void* allocate(size_t byte_size) const override;

    void release(void* ptr) const override;
};