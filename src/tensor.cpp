#include "layer/tensor.h"

inline size_t data_type_size(DataType data_type) {
    switch (data_type) {
        case DataType::kDataTypeFp32: {
            return 4;
        }
        case DataType::kDataTypeFp16: {
            return 2;
        }
        case DataType::kDataTypeInt8: {
            return 1;
        }
        case DataType::kDataTypeInt32: {
            return 4;
        }
        default: {
            return 0;
        }
    }
}

namespace tensor{

Tensor::Tensor(DataType data_type, int32_t dim0, bool need_alloc,
            std::shared_ptr<DeviceAllocator> alloc, void* ptr) : 
    data_type_(data_type) {
    dims_.push_back(dim0);
    size_ = dim0;
    byte_size_ = size_ * data_type_size(data_type);
    if (need_alloc && alloc) {
        buffer_ = std::make_shared<hpinfer::Buffer>(byte_size, alloc);
    } else {
        buffer_ = std::make_shared<hpinfer::Buffer>(byte_size, nullptr, ptr, true);
    }
}

Tensor::Tensor(DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
            std::shared_ptr<DeviceAllocator> alloc, void* ptr) : 
    data_type_(data_type) {
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    size_ = dim0 * dim1;
    byte_size_ = size_ * data_type_size(data_type);
    if (need_alloc && alloc) {
        buffer_ = std::make_shared<hpinfer::Buffer>(byte_size, alloc);
    } else {
        buffer_ = std::make_shared<hpinfer::Buffer>(byte_size, nullptr, ptr, true);
    }
}

Tensor::Tensor(DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc,
            std::shared_ptr<DeviceAllocator> alloc, void* ptr) : 
    data_type_(data_type) {
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    size_ = dim0 * dim1 * dim2;
    byte_size_ = size_ * data_type_size(data_type);
    if (need_alloc && alloc) {
        buffer_ = std::make_shared<hpinfer::Buffer>(byte_size, alloc);
    } else {
        buffer_ = std::make_shared<hpinfer::Buffer>(byte_size, nullptr, ptr, true);
    }
}

Tensor::Tensor(DataType data_type, std::vector<int32_t> dims, bool need_alloc,
            std::shared_ptr<DeviceAllocator> alloc, void* ptr) : 
    data_type_(data_type), dims_(std::move(dims)) {
    size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<>());
    byte_size_ = size_ * data_type_size(data_type);
    if (need_alloc && alloc) {
        buffer_ = std::make_shared<hpinfer::Buffer>(byte_size, alloc);
    } else {
        buffer_ = std::make_shared<hpinfer::Buffer>(byte_size, nullptr, ptr, true);
    }
}

void Tensor::init_buffer(std::shared_ptr<DeviceAllocator> alloc, DataType data_type,
                        bool need_alloc, void* ptr) {
    if (!alloc && !need_alloc) {
        std::shared_ptr<hpinfer::Buffer> buffer =
        std::make_shared<hpinfer::Buffer>(data_type_size(data_type) * size_, nullptr, ptr, true);
        this->buffer_ = buffer;
    } else {
        std::cerr << "The allocator parameter in the allocate function is not null or \
        need_alloc is true" << std::endl;
    }
}
}
