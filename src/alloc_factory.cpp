#include "base/alloc_factory.h"


std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
std::once_flag CPUDeviceAllocatorFactory::initInstanceFlag;

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;
std::once_flag CUDADeviceAllocatorFactory::initInstanceFlag;

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::get_instance() {
    if (instance == nullptr) {
        std::call_once(initInstanceFlag, []() {
            instance = std::make_shared<CPUDeviceAllocator>();});
    }
    return instance;
}

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::get_instance() {
    if (instance == nullptr) {
        std::call_once(initInstanceFlag, []() {
            instance = std::make_shared<CUDADeviceAllocator>();});
    }
    return instance;
}