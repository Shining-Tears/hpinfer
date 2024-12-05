#pragma once

#include "layer/layer.h"

namespace op {
class VecAddLayer : public Layer {
public:
  explicit VecAddLayer(DeviceType device_type);

  void forward() override;
};
}