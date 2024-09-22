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

void Tensor::to_cpu(void* stream) {
    DeviceType device_type = this->device_type();

    if (device_type == DeviceType::kDeviceCUDA) {
        size_t byte_size = this->byte_size();
        auto cpu_alloc = CPUDeviceAllocatorFactory::get_instance();
        auto cpu_buffer = std::make_shared<hpinfer::Buffer>(byte_size, cpu_alloc);
        cpu_alloc->memcpy(buffer_->ptr(), cpu_buffer->ptr(), byte_size,
                      MemcpyKind::kMemcpyCUDA2CPU, static_cast<cudaStream_t>(stream), true);
        this->buffer_ = cpu_buffer;
    } else {
        std::cerr << "The device type of the tensor is not cuda.";
    }
}

void Tensor::to_cuda(void* stream) {
    DeviceType device_type = this->device_type();

    if (device_type == DeviceType::kDeviceCPU) {
        size_t byte_size = this->byte_size();
        auto cuda_alloc = CUDADeviceAllocatorFactory::get_instance();
        auto cuda_buffer = std::make_shared<hpinfer::Buffer>(byte_size, cuda_alloc);
        cuda_alloc->memcpy(buffer_->ptr(), cuda_buffer->ptr(), byte_size,
                      MemcpyKind::kMemcpyCUDA2CPU, static_cast<cudaStream_t>(stream), true);
        this->buffer_ = cuda_buffer;
    } else {
        std::cerr << "The device type of the tensor is not cpu.";
    }
}

bool Tensor::is_empty() const {
    return size_ == 0 || buffer_ == nullptr;
}

void Tensor::reshape(const std::vector<int32_t>& dims) {
    size_t size = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<>());
    this->byte_size_ = size * data_type_size(this->data_type());

    if (size > size_) {
        auto new_buffer = std::make_shared<hpinfer::Buffer>(size * data_type_size(this->data_type_),
                                                     buffer_->allocator());
        new_buffer->copy_from(buffer_.get());
        this->buffer_ = new_buffer;
    }
    this->dims_ = dims;
    this->size_ = size;
}

size_t Tensor::size() const { return this->size_; }

size_t Tensor::byte_size() const { return this->byte_size_; }

DataType Tensor::data_type() const { return this->data_type_; }

const std::vector<int32_t>& Tensor::dims() const { return this->dims_; }

DeviceType Tensor::device_type() const { return this->buffer_->device_type(); };

void Tensor::reset(DataType data_type, const std::vector<int32_t>& dims) {
  this->data_type_ = data_type;
  this->dims_ = dims;
  this->size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<>());
  this->buffer_ = nullptr;
}

Tensor Tensor::clone() const {
  Tensor new_tensor = *this;
  size_t byte_size = this->byte_size();

  auto allocator = buffer_->allocator();
  new_tensor.buffer_ = std::make_shared<hpinfer::Buffer>(byte_size, allocator);
  new_tensor.buffer_->copy_from(buffer_.get());
  return new_tensor;
}

template <typename T>
T& Tensor::index(int64_t offset) {
    if (offset < 0 || offset > size_) {
        std::cerr << "The offeset is illegal."
    }
    T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
    return val;
}

template <typename T>
const T& Tensor::index(int64_t offset) const {
    if (offset < 0 || offset > size_) {
        std::cerr << "The offeset is illegal."
    }
    const T& val = *(reinterpret_cast<T*>(buffer_->ptr()) + offset);
    return val;
}

template <typename T>
const T* Tensor::ptr() const {
    if (!buffer_) {
        return nullptr;
    }
    return reinterpret_cast<const T*>(buffer_->ptr());
}

template <typename T>
T* Tensor::ptr() {
    if (!buffer_) {
        return nullptr;
    }
    return reinterpret_cast<T*>(buffer_->ptr());
}

template <typename T>
T* Tensor::ptr(int64_t index) {
    if (!buffer_) {
        return nullptr;
    }
    return reinterpret_cast<T*>(buffer_->ptr()) + index;
}

template <typename T>
const T* Tensor::ptr(int64_t index) const {
    if (!buffer_) {
            return nullptr;
    }
    return reinterpret_cast<const T*>(buffer_->ptr()) + index;
}
}
