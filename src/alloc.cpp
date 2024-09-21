#include "base/alloc.h"
#include <cuda_runtime_api.h>
#include <iostream>

void DeviceAllocator::memcpy(void* dst, const void* src, size_t byte_size, MemcpyKind memcpy_kind, 
                        void* stream, bool need_sync) const
{
    if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
        std::memcpy(dst, src, byte_size);
    } else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
        if (need_sync && stream != nullptr) {
            cudaMemcpyAsync(dst, src, byte_size, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream));
        } else {
            cudaMemcpy(dst, src, byte_size, cudaMemcpyHostToDevice);
        }
    } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
        if (need_sync && stream != nullptr) {
            cudaMemcpyAsync(dst, src, byte_size, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream));
        } else {
            cudaMemcpy(dst, src, byte_size, cudaMemcpyDeviceToDevice);
        }
    } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
        if (need_sync && stream != nullptr) {
            cudaMemcpyAsync(dst, src, byte_size, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream));
        } else {
            cudaMemcpy(dst, src, byte_size, cudaMemcpyDeviceToHost);
        }
    } else {
        std::cerr << "Unknow memcpy kind. \n";
    }
}                           

CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {};

void* CPUDeviceAllocator::allocate(size_t byte_size) const {
    if (byte_size > 0) {
        void* data = malloc(byte_size);
        return data;
    } else {
        return nullptr;
    }
}

void CPUDeviceAllocator::release(void* ptr) const {
    if (ptr != nullptr) {
        free(ptr);
    }
}

// 派生类构造函数中初始化基类部分应该在构造函数的初始化列表中进行
CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {};

void* CUDADeviceAllocator::allocate(size_t byte_size) const {
    if (byte_size > 0) {
        void* cuda_ptr = nullptr;
        cudaMalloc(&cuda_ptr, byte_size);
        return cuda_ptr;
    }
}

void CUDADeviceAllocator::release(void* ptr) const {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}