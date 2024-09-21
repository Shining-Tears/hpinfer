#pragma once

#include <cuda_runtime_api.h>
#include <armadillo>
#include <memory>
#include <vector>
#include "base/buffer.h"
#include "layer/layer.h"

enum class DataType {
  kDataTypeUnknown = 0,
  kDataTypeFp32 = 1,
  kDataTypeFp16 = 2,
  kDataTypeInt8 = 3,
  kDataTypeInt32 = 4,
};

enum class LayerType {
  kLayerUnknown = 0,
  kLayerLinear = 1,
  kLayerEncode = 2,
  kLayerEmbedding = 3,
  kLayerRMSNorm = 4,
  kLayerMatmul = 5,
  kLayerRoPe = 6,
  kLayerMHA = 7,
  kLayerSoftmax = 8,
  kLayerAdd = 9,
  kLayerSwiGLU = 10,
};

enum class ModelType : uint8_t {
  kModelTypeUnknown = 0,
  kModelTypeLLama2 = 1,
};

namespace tensor {

class Tensor {
    size_t size_ = 0;
    size_t byte_size_ = 0;
    std::vector<int32_t> dims_;
    std::shared_ptr<hpinfer::Buffer> buffer_;
    DataType data_type_ = DataType::kDataTypeUnknown;

public:
    explicit Tensor() = default;

    explicit Tensor(DataType data_type, int32_t dim0, bool need_alloc = false,
                std::shared_ptr<DeviceAllocator> alloc = nullptr, void* ptr = nullptr);
    
    explicit Tensor(DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
                std::shared_ptr<DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

    explicit Tensor(DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc = false,
                std::shared_ptr<DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

    explicit Tensor(DataType data_type, std::vector<int32_t> dims, bool need_alloc = false,
                std::shared_ptr<DeviceAllocator> alloc = nullptr, void* ptr = nullptr);

    void to_cpu();

    void to_cuda();

    bool is_empty() const;

    void init_buffer(std::shared_ptr<DeviceAllocator> alloc, DataType data_type, bool need_alloc = false,
                void* ptr);
    
    bool assign(std::shared_ptr<hpinfer::Buffer> buffer);

    template <typename T>
    T* ptr();

    template <typename T>
    const T* ptr() const;

    void reshape(const std::vector<int32_t>& dims);

    void reset(DataType data_type, const std::vector<int32_t>& dims);

    std::shared_ptr<hpinfer::Buffer> get_buffer() const;
    
    size_t size() const;

    size_t byte_size() const;
    
    DataType data_type() const;

    void set_device_type(DeviceType device_type) const;

    const std::vector<int32_t>& dims() const;

    DeviceType device_type() const;

    template <typename T>
    T* ptr(int64_t index);

    template <typename T>
    const T* ptr(int64_t index) const;

    template <typename T>
    T& index(int64_t offset);

    template <typename T>
    const T& index(int64_t offset) const;

    tensor::Tensor clone() const;
    
};
}