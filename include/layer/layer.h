#pragma once

#include "base/alloc.h"
#include "base/cuda_config.h"
#include "layer/tensor.h"

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

class Layer: public BaseLayer {
public:
  explicit Layer(DeviceType device_type, LayerType layer_type, DataType data_type,
                     std::string layer_name = "");

  void forward() override;

  void forward(const tensor::Tensor& input1, const tensor::Tensor& output1) override;

  void forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& output1) override;

  void forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& output1) override;

  void forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& input4,
                               const tensor::Tensor& output1) override;

  void forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                               const tensor::Tensor& input3, const tensor::Tensor& input4,
                               const tensor::Tensor& input5, const tensor::Tensor& output1) override;

  void set_input(int32_t idx, const tensor::Tensor& input) override;

  void set_output(int32_t idx, const tensor::Tensor& output) override;

  void set_input_size(size_t size);

  void set_output_size(size_t size);
  
  size_t input_size() const override;

  size_t output_size() const override;

  tensor::Tensor& get_input(int32_t idx) override;

  tensor::Tensor& get_output(int32_t idx) override;

  const tensor::Tensor& get_input(int32_t idx) const override;

  const tensor::Tensor& get_output(int32_t idx) const override;

  virtual void to_cuda();
  
  void set_cuda_config(std::shared_ptr<kernel::CudaConfig> config);

  std::shared_ptr<hpinfer::CudaConfig> cuda_config() const;

protected:
  std::vector<tensor::Tensor> inputs_;
  std::vector<tensor::Tensor> outputs_;
  std::shared_ptr<hpinfer::CudaConfig> cuda_config_;
};

class LayerParam : public Layer {
public:
  explicit LayerParam(DeviceType device_type, LayerType layer_type, DataType data_type,
                      bool is_quant_layer = false, std::string layer_name = "");

  size_t weight_size() const;

  tensor::Tensor& get_weight(int32_t idx);

  const tensor::Tensor& get_weight(int32_t idx) const;

  void to_cuda() override;

  void set_weight_size(size_t size);

  void set_weight(int32_t idx, const tensor::Tensor& weight);

  void set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
                          DeviceType device_type = DeviceType::kDeviceUnknown);

  void set_scales(const tensor::Tensor& scales);

  void set_group_size(int32_t group_size);

  int32_t get_scale_num() const;

protected:
  int32_t group_size_ = 0;
  bool is_quant_layer_ = false;
  tensor::Tensor scales_;
  std::vector<tensor::Tensor> weights_;
};
}