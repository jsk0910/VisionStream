#include "core/memory/vision_buffer.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("CUDA Error"); \
        } \
    } while (0)

namespace visionstream {

int get_element_size(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT16: return 2;
        case DataType::INT8: return 1;
        case DataType::UINT8: return 1;
        case DataType::INT32: return 4;
        case DataType::INT64: return 8;
        default: throw std::invalid_argument("Unknown DataType");
    }
}

VisionBuffer::VisionBuffer(const TensorShape& shape, DataType dtype, DeviceType device, int device_id)
    : shape_(shape), dtype_(dtype), device_(device), device_id_(device_id), data_ptr_(nullptr), capacity_bytes_(0) {
    allocate();
}

VisionBuffer::~VisionBuffer() {
    deallocate();
}

VisionBuffer::VisionBuffer(VisionBuffer&& other) noexcept
    : shape_(std::move(other.shape_)), dtype_(other.dtype_),
      device_(other.device_), device_id_(other.device_id_),
      data_ptr_(other.data_ptr_), capacity_bytes_(other.capacity_bytes_) {
    other.data_ptr_ = nullptr;
    other.capacity_bytes_ = 0;
}

VisionBuffer& VisionBuffer::operator=(VisionBuffer&& other) noexcept {
    if (this != &other) {
        deallocate();
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        device_ = other.device_;
        device_id_ = other.device_id_;
        data_ptr_ = other.data_ptr_;
        capacity_bytes_ = other.capacity_bytes_;
        
        other.data_ptr_ = nullptr;
        other.capacity_bytes_ = 0;
    }
    return *this;
}

size_t VisionBuffer::size_bytes() const {
    return shape_.num_elements() * get_element_size(dtype_);
}

void VisionBuffer::allocate() {
    size_t size = size_bytes();
    if (size == 0) return;

    if (device_ == DeviceType::CUDA) {
        int current_device;
        CUDA_CHECK(cudaGetDevice(&current_device));
        if (current_device != device_id_) {
            CUDA_CHECK(cudaSetDevice(device_id_));
        }
        CUDA_CHECK(cudaMalloc(&data_ptr_, size));
        if (current_device != device_id_) {
            CUDA_CHECK(cudaSetDevice(current_device));
        }
    } else {
        data_ptr_ = std::malloc(size);
        if (!data_ptr_) {
            throw std::bad_alloc();
        }
    }
    capacity_bytes_ = size;
}

void VisionBuffer::deallocate() {
    if (data_ptr_ != nullptr) {
        if (device_ == DeviceType::CUDA) {
            int current_device;
            cudaGetDevice(&current_device);
            if (current_device != device_id_) {
                cudaSetDevice(device_id_);
            }
            cudaFree(data_ptr_);
            if (current_device != device_id_) {
                cudaSetDevice(current_device);
            }
        } else {
            std::free(data_ptr_);
        }
        data_ptr_ = nullptr;
        capacity_bytes_ = 0;
    }
}

void VisionBuffer::copy_to_host(void* dst) const {
    if (size_bytes() == 0) return;
    
    if (device_ == DeviceType::CUDA) {
        CUDA_CHECK(cudaMemcpy(dst, data_ptr_, size_bytes(), cudaMemcpyDeviceToHost));
    } else {
        std::memcpy(dst, data_ptr_, size_bytes());
    }
}

void VisionBuffer::copy_to_device(void* dst, int dst_device_id) const {
    if (size_bytes() == 0) return;

    if (device_ == DeviceType::CUDA) {
        if (device_id_ == dst_device_id) {
            CUDA_CHECK(cudaMemcpy(dst, data_ptr_, size_bytes(), cudaMemcpyDeviceToDevice));
        } else {
            CUDA_CHECK(cudaMemcpyPeer(dst, dst_device_id, data_ptr_, device_id_, size_bytes()));
        }
    } else {
        CUDA_CHECK(cudaMemcpy(dst, data_ptr_, size_bytes(), cudaMemcpyHostToDevice));
    }
}

std::shared_ptr<VisionBuffer> VisionBuffer::clone_to(DeviceType target_device, int target_device_id) const {
    auto new_buf = std::make_shared<VisionBuffer>(shape_, dtype_, target_device, target_device_id);
    
    if (size_bytes() > 0) {
        if (target_device == DeviceType::CPU) {
            copy_to_host(new_buf->data());
        } else {
            copy_to_device(new_buf->data(), target_device_id);
        }
    }
    
    return new_buf;
}

} // namespace visionstream
