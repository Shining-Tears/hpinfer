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
}
