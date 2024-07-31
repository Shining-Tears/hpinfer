#pragma once

#include "base/alloc.h"
#include "base/alloc_factory.h"

namespace hpinfer {
class Buffer : std::enable_shared_from_this<Buffer> {
    size_t byte_size_ = 0;
    void* ptr_ = nullptr;
    bool use_external_ = false;
    DeviceType device_type_ = DeviceType::kDeviceUnkown;
    std::shared_ptr<DeviceAllocator> allocator_;

public:
    explicit Buffer() = default;

    explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                    void* ptr = nullptr, bool use_external = false);

    Buffer(const Buffer&) = delete;
    Buffer &operator = (const Buffer&) = delete;

    virtual ~Buffer();

    bool allocate();

    void copy_from(const Buffer& buffer) const;

    void copy_from(const Buffer* buffer) const;

    void* ptr();

    const void* ptr() const;

    size_t byte_size() const;

    std::shared_ptr<DeviceAllocator> allocator() const;

    DeviceType device_type() const;

    void set_device_type(DeviceType device_type);

    std::shared_ptr<Buffer> get_shared_from_this();

    bool is_external() const;
};
}