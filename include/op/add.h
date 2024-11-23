#pragma once

#include "base/base.h"
#include "layer/layer.h"

namespace hpinfer {
class VecAddLayer : public Layer {
 public:
  explicit VecAddLayer(DeviceType device_type);

  void check() const override;

  void forward() override;
};
}