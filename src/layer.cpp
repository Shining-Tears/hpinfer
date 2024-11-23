#include "layer/layer.h"

namespace hpinfer {

BaseLayer::BaseLayer(DeviceType device_type, LayerType layer_type, DataType data_type,
                     std::string layer_name)
    : device_type_(device_type),
      layer_type_(layer_type),
      data_type_(data_type),
      layer_name_(layer_name) {}

DataType BaseLayer::data_type() const { return data_type_; }

LayerType BaseLayer::layer_type() const { return layer_type_; }

DeviceType BaseLayer::device_type() const { return device_type_; }

const std::string& BaseLayer::layer_name() const { return layer_name_; }

void BaseLayer::set_data_type(DataType data_type) { data_type_ = data_type; }

void BaseLayer::set_layer_type(LayerType layer_type) { layer_type_ = layer_type; }

void BaseLayer::set_device_type(DeviceType device_type) { device_type_ = device_type; }

void BaseLayer::set_layer_name(const std::string& layer_name) { layer_name_ = layer_name; }

Layer::Layer(DeviceType device_type, LayerType layer_type, DataType data_type, std::string layer_name)
    : BaseLayer(device_type, layer_type, data_type, std::move(layer_name)) {}

void Layer::set_input(int32_t idx, const tensor::Tensor& input) { inputs_.at(idx) = input; }

void Layer::set_output(int32_t idx, const tensor::Tensor& output) { outputs_.at(idx) = output; }

void Layer::set_input_size(size_t size) { inputs_.resize(size); }

void Layer::set_output_size(size_t size) { outputs_.resize(size); }

size_t Layer::input_size() const { return inputs_.size(); }

size_t Layer::output_size() const { return outputs_.size(); }

const tensor::Tensor& Layer::get_input(int32_t idx) const { return inputs_.at(idx); }

tensor::Tensor& Layer::get_input(int32_t idx) { return inputs_.at(idx);}

tensor::Tensor& Layer::get_output(int32_t idx) { return outputs_.at(idx); }

const tensor::Tensor& Layer::get_output(int32_t idx) const { return outputs_.at(idx);}

void Layer::to_cuda() {
  for (auto& input : inputs_) {
    if (!input.is_empty()) {
      input.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
  for (auto& output : outputs_) {
    if (!output.is_empty()) {
      output.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
}

void Layer::set_cuda_config(std::shared_ptr<hpinfer::CudaConfig> config) { cuda_config_ = config;}

std::shared_ptr<hpinfer::CudaConfig> Layer::cuda_config() const { return cuda_config_; }

void LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) { weights_.at(idx) = weight; }

const tensor::Tensor& LayerParam::get_weight(int32_t idx) const { return weights_.at(idx);}

void LayerParam::to_cuda() {
  Layer::to_cuda();
  for (auto& weight : weights_) {
    weight.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
  }
  if (!scales_.is_empty()) {
    scales_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
  }
}

void LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                    const void* weight_ptr, DeviceType device_type) {
  
  size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<>());
  std::shared_ptr<hpinfer::Buffer> buffer =
      std::make_shared<hpinfer::Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
  if (device_type != DeviceType::kDeviceUnknown) {
    buffer->set_device_type(device_type);
  }

  if (!is_quant_layer_) {
    tensor::Tensor weight(DataType::kDataTypeFp32, dims);
    weight.set_device_type(device_type);
    weights_.at(idx) = weight;
  } else {
    // is quant layer
    tensor::Tensor weight(DataType::kDataTypeInt8, dims);
    weight.set_device_type(device_type);
    weights_.at(idx) = weight;

    const int32_t weight_size = static_cast<int32_t>(weight.size());

    int32_t scale_nums = weight_size / group_size_;
    scales_ = tensor::Tensor{DataType::kDataTypeFp32, scale_nums, false, nullptr,
                             reinterpret_cast<float*>((int8_t*)weight_ptr + weight_size)};
    scales_.set_device_type(device_type);
  }
}

void LayerParam::set_scales(const tensor::Tensor& scales) { scales_ = scales;}

void LayerParam::set_group_size(int32_t group_size) { group_size_ = group_size; }

int32_t LayerParam::get_scale_num() const { return static_cast<int32_t>(scales_.size());}

void LayerParam::set_weight_size(size_t size) { weights_.resize(size); }

size_t LayerParam::weight_size() const { return weights_.size(); }

void Layer::forward() {};

void Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_output(0, output1);
  return this->forward();
}

void Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);

  this->set_output(0, output1);
  return this->forward();
}

void Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);

  this->set_output(0, output1);
  return this->forward();
}

void Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);

  this->set_output(0, output1);
  return this->forward();
}

void Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                            const tensor::Tensor& input3, const tensor::Tensor& input4,
                            const tensor::Tensor& input5, const tensor::Tensor& output1) {
  this->set_input(0, input1);
  this->set_input(1, input2);
  this->set_input(2, input3);
  this->set_input(3, input4);
  this->set_input(4, input5);

  this->set_output(0, output1);
  return this->forward();
}

tensor::Tensor& LayerParam::get_weight(int32_t idx) { return weights_.at(idx); }
}
