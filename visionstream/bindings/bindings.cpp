#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "core/memory/vision_buffer.h"
#include "core/graph/node.h"
#include "core/graph/graph_manager.h"
#include "core/codec/entropy_model.h"
#include "core/codec/arithmetic_coder.h"

namespace py = pybind11;
using namespace visionstream;

// Trampoline class for Node to allow Python inheritance
class PyNode : public Node {
public:
    using Node::Node; // Inherit constructors

    std::shared_ptr<VisionBuffer> process(std::shared_ptr<VisionBuffer> input) override {
        PYBIND11_OVERRIDE_PURE(
            std::shared_ptr<VisionBuffer>, // Return type
            Node,                          // Parent class
            process,                       // Name of function in C++
            input                          // Arguments
        );
    }
};

PYBIND11_MODULE(visionstream, m) {
    m.doc() = "VisionStream Hybrid Vision Research Framework Core";

    py::enum_<DataType>(m, "DataType")
        .value("FLOAT32", DataType::FLOAT32)
        .value("FLOAT16", DataType::FLOAT16)
        .value("INT8", DataType::INT8)
        .value("UINT8", DataType::UINT8)
        .value("INT32", DataType::INT32)
        .value("INT64", DataType::INT64)
        .export_values();

    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA)
        .export_values();

    py::class_<TensorShape>(m, "TensorShape")
        .def(py::init<>())
        .def(py::init<std::vector<int64_t>>())
        .def_readwrite("dims", &TensorShape::dims)
        .def("num_elements", &TensorShape::num_elements);

    py::class_<VisionBuffer, std::shared_ptr<VisionBuffer>>(m, "VisionBuffer", py::buffer_protocol())
        .def(py::init<const TensorShape&, DataType, DeviceType, int>(), 
             py::arg("shape"), py::arg("dtype"), py::arg("device"), py::arg("device_id") = 0)
        .def_property_readonly("shape", [](const VisionBuffer& vb) { return vb.shape().dims; })
        .def_property_readonly("dtype", &VisionBuffer::dtype)
        .def_property_readonly("device", &VisionBuffer::device)
        .def_property_readonly("device_id", &VisionBuffer::device_id)
        .def_property_readonly("size_bytes", &VisionBuffer::size_bytes)
        .def("clone_to", &VisionBuffer::clone_to, py::arg("target_device"), py::arg("target_device_id") = 0)
        // Add numpy buffer interface for CPU buffers
        .def_buffer([](VisionBuffer& vb) -> py::buffer_info {
            if (vb.device() != DeviceType::CPU) {
                throw std::runtime_error("Only CPU buffers support the numpy buffer protocol directly.");
            }
            std::string format;
            int itemsize = get_element_size(vb.dtype());
            switch (vb.dtype()) {
                case DataType::FLOAT32: format = py::format_descriptor<float>::format(); break;
                // Add more format descriptors later (FLOAT16 is tricky without numpy-ctype, just basic for now)
                case DataType::UINT8: format = py::format_descriptor<uint8_t>::format(); break;
                case DataType::INT32: format = py::format_descriptor<int32_t>::format(); break;
                case DataType::INT64: format = py::format_descriptor<int64_t>::format(); break;
                default: throw std::runtime_error("Unsupported dtype for numpy buffer protocol.");
            }

            // Calculate strides for C-contiguous
            std::vector<py::ssize_t> strides(vb.shape().dims.size());
            py::ssize_t stride = itemsize;
            for (int i = vb.shape().dims.size() - 1; i >= 0; i--) {
                strides[i] = stride;
                stride *= vb.shape().dims[i];
            }

            return py::buffer_info(
                vb.data(),                              // Pointer to buffer
                itemsize,                               // Size of one scalar
                format,                                 // Python struct-style format descriptor
                vb.shape().dims.size(),                 // Number of dimensions
                vb.shape().dims,                        // Buffer dimensions
                strides                                 // Strides (in bytes) for each index
            );
        })
        // Helper to get raw pointer as integer (e.g. for PyTorch dlpack / cuda memory wrapping)
        .def("data_ptr", [](const VisionBuffer& vb) {
            return reinterpret_cast<uintptr_t>(vb.data());
        });

    py::class_<Node, PyNode, std::shared_ptr<Node>>(m, "Node")
        .def(py::init<const std::string&>(), py::arg("name"))
        .def("process", &Node::process)
        .def_property_readonly("name", &Node::name)
        .def_property("is_bypassed", &Node::is_bypassed, &Node::set_bypassed)
        .def_property_readonly("last_latency_ms", &Node::last_latency_ms);

    py::class_<GraphManager>(m, "GraphManager")
        .def(py::init<>())
        .def("add_node", &GraphManager::add_node)
        .def("get_node", &GraphManager::get_node)
        .def("execute", &GraphManager::execute)
        .def("get_latencies", &GraphManager::get_latencies);

    py::class_<ArithmeticCoder>(m, "ArithmeticCoder")
        .def(py::init<>())
        .def_static("encode", [](const std::vector<int16_t>& symbols, const std::vector<int16_t>& indexes, const std::vector<int32_t>& cdfs, const std::vector<int32_t>& cdf_sizes, const std::vector<int32_t>& offsets, int precision) {
            std::string s = ArithmeticCoder::encode(symbols, indexes, cdfs, cdf_sizes, offsets, precision);
            return py::bytes(s);
        }, py::arg("symbols"), py::arg("indexes"), py::arg("cdfs"), py::arg("cdf_sizes"), py::arg("offsets"), py::arg("precision") = 16)
        .def_static("decode", [](py::bytes bitstream, const std::vector<int16_t>& indexes, const std::vector<int32_t>& cdfs, const std::vector<int32_t>& cdf_sizes, const std::vector<int32_t>& offsets, int precision) {
            std::string stream_str = bitstream;
            return ArithmeticCoder::decode(stream_str, indexes, cdfs, cdf_sizes, offsets, precision);
        }, py::arg("bitstream"), py::arg("indexes"), py::arg("cdfs"), py::arg("cdf_sizes"), py::arg("offsets"), py::arg("precision") = 16);
}
