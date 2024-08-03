#pragma once

#include "base/alloc.h"

enum class DataType {
  kDataTypeUnknown = 0,
  kDataTypeFp32 = 1,
  kDataTypeFp16 = 1,
  kDataTypeInt8 = 2,
  kDataTypeInt32 = 3,
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

namespace hpinfer {
class BaseLayer {
 public:
  explicit BaseLayer(DeviceType device_type, LayerType layer_type, DataType data_type,
                     std::string layer_name = "");

  DataType data_type() const;

  LayerType layer_type() const;

  DeviceType device_type() const;

  const std::string& layer_name() const;

  void set_data_type(DataType data_type);

  void set_layer_type(LayerType layer_type);

  void set_device_type(DeviceType device_type);

  void set_layer_name(const std::string& layer_name);

  virtual void init() = 0;

  virtual void forward() = 0;

  virtual void forward(const tensor::Tensor& input1, const tensor::Tensor& output1) = 0;

  virtual void forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& output1) = 0;

  virtual void forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& output1) = 0;

  virtual void forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& input4,
                               const tensor::Tensor& output1) = 0;

  virtual void forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& input4,
                               const tensor::Tensor& input5, const tensor::Tensor& output1) = 0;

  virtual void set_input(int32_t idx, const tensor::Tensor& input) = 0;

  virtual void set_output(int32_t idx, const tensor::Tensor& output) = 0;

  virtual size_t input_size() const = 0;

  virtual size_t output_size() const = 0;

  virtual void check() const = 0;

  virtual tensor::Tensor& get_input(int32_t idx) = 0;

  virtual tensor::Tensor& get_output(int32_t idx) = 0;

  virtual const tensor::Tensor& get_input(int32_t idx) const = 0;

  virtual const tensor::Tensor& get_output(int32_t idx) const = 0;

 protected:
  std::string layer_name_;
  LayerType layer_type_ = LayerType::kLayerUnknown;
  DataType data_type_ = DataType::kDataTypeUnknown;
  DeviceType device_type_ = DeviceType::kDeviceUnknown;
};
}