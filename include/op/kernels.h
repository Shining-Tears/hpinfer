#pragma once

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include "layer/layer.h"

#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])

using AddKernel = void (*)(const tensor::Tensor& input1, const tensor::Tensor& input2,
                           const tensor::Tensor& output, void* stream);

using RMSKernel = void (*)(const tensor::Tensor& input, const tensor::Tensor& weight,
                              const tensor::Tensor& output, void* stream);