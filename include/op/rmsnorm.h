#pragma once

#include "layer/layer.h"

namespace op {
class RmsNormLayer : public LayerParam {
  int32_t dim_ = 0;
public:
  explicit RmsNormLayer(DeviceType device_type, int32_t dim);

  void forward() override;
};
}