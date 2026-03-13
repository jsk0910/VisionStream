#ifndef VISIONSTREAM_CORE_MEMORY_VISION_BUFFER_H
#define VISIONSTREAM_CORE_MEMORY_VISION_BUFFER_H

#include <vector>
#include <string>
#include <stdexcept>
#include <memory>

namespace visionstream {

enum class DataType {
    FLOAT32,
    FLOAT16,
    INT8,
    UINT8,
    INT32,
    INT64
};

enum class DeviceType {
    CPU,
    CUDA
};

struct TensorShape {
    std::vector<int64_t> dims;
    
    TensorShape() = default;
    TensorShape(const std::vector<int64_t>& d) : dims(d) {}
    
    int64_t num_elements() const {
        if (dims.empty()) return 0;
        int64_t count = 1;
        for (auto d : dims) {
            count *= d;
        }
        return count;
    }
};

class VisionBuffer {
public:
    VisionBuffer(const TensorShape& shape, DataType dtype, DeviceType device, int device_id = 0);
    ~VisionBuffer();

    // Prevent copying to maintain zero-copy where possible
    VisionBuffer(const VisionBuffer&) = delete;
    VisionBuffer& operator=(const VisionBuffer&) = delete;

    // Allow moving
    VisionBuffer(VisionBuffer&& other) noexcept;
    VisionBuffer& operator=(VisionBuffer&& other) noexcept;

    // Accessors
    void* data() const { return data_ptr_; }
    const TensorShape& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    DeviceType device() const { return device_; }
    int device_id() const { return device_id_; }
    size_t size_bytes() const;

    // Data Transfer (Synchronous for now)
    void copy_to_host(void* dst) const;
    void copy_to_device(void* dst, int dst_device_id) const;
    
    // Create a deep copy on a specific device
    std::shared_ptr<VisionBuffer> clone_to(DeviceType target_device, int target_device_id = 0) const;

private:
    void allocate();
    void deallocate();

    TensorShape shape_;
    DataType dtype_;
    DeviceType device_;
    int device_id_;
    void* data_ptr_;
    size_t capacity_bytes_;
};

int get_element_size(DataType dtype);

} // namespace visionstream

#endif // VISIONSTREAM_CORE_MEMORY_VISION_BUFFER_H
