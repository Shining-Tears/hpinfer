#pragma once

#include "layer/tensor.h"

namespace op {
  void add_kernel_cu(const tensor::Tensor& input1, const tensor::Tensor& input2,
  const tensor::Tensor& output, void* stream = nullptr);
}

