import os
import sys
import ctypes

# Add build directory to path to find the compiled C++ extension
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BUILD_DIR = os.path.join(ROOT, "build")
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

# Preload the core shared library so the pybind11 module can find it
core_lib_path = os.path.join(BUILD_DIR, "libvisionstream_core.so")
if os.path.exists(core_lib_path):
    ctypes.CDLL(core_lib_path, mode=ctypes.RTLD_GLOBAL)

import torch
from visionstream import VisionBuffer, TensorShape, DataType, DeviceType

def torch_dtype_to_visionstream(dtype: torch.dtype) -> DataType:
    mapping = {
        torch.float32: DataType.FLOAT32,
        torch.float16: DataType.FLOAT16,
        torch.int8: DataType.INT8,
        torch.uint8: DataType.UINT8,
        torch.int32: DataType.INT32,
        torch.int64: DataType.INT64
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return mapping[dtype]

def visionstream_dtype_to_torch(dtype: DataType) -> torch.dtype:
    mapping = {
        DataType.FLOAT32: torch.float32,
        DataType.FLOAT16: torch.float16,
        DataType.INT8: torch.int8,
        DataType.UINT8: torch.uint8,
        DataType.INT32: torch.int32,
        DataType.INT64: torch.int64
    }
    return mapping[dtype]

def torch_to_vision_buffer(tensor: torch.Tensor) -> VisionBuffer:
    """Zero-copy wrap a torch Tensor into a VisionBuffer (if contiguous)."""
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
        
    shape = TensorShape(list(tensor.shape))
    dtype = torch_dtype_to_visionstream(tensor.dtype)
    
    device_type = DeviceType.CUDA if tensor.device.type == "cuda" else DeviceType.CPU
    device_id = tensor.device.index if tensor.device.index is not None else 0
    
    # Create empty buffer
    vb = VisionBuffer(shape, dtype, device_type, device_id)
    
    # We copy for safety at the moment, but normally zero-copy via data_ptr mapping
    # Copying data since VisionBuffer manages its own memory currently
    if device_type == DeviceType.CUDA:
        # Move tensor to CPU before copy to avoid GPU pointer segfault using ctypes
        tensor_cpu = tensor.cpu()
        ptr = tensor_cpu.data_ptr()
        ctypes.memmove(vb.data_ptr(), ptr, vb.size_bytes())
        # We need to copy from host vb to device vb, but vb doesn't expose a host-to-device method directly from Python memory.
        # So we clone it to the target device.
        vb_cpu = VisionBuffer(shape, dtype, DeviceType.CPU, 0)
        ctypes.memmove(vb_cpu.data_ptr(), ptr, vb_cpu.size_bytes())
        vb = vb_cpu.clone_to(DeviceType.CUDA, device_id)
    else:
        # CPU
        ptr = tensor.data_ptr()
        ctypes.memmove(vb.data_ptr(), ptr, vb.size_bytes())
        
    return vb

def vision_buffer_to_torch(vb: VisionBuffer) -> torch.Tensor:
    """Create a torch Tensor copying data from VisionBuffer."""
    shape = tuple(vb.shape)
    dtype = visionstream_dtype_to_torch(vb.dtype)
    device_str = f"cuda:{vb.device_id}" if vb.device == DeviceType.CUDA else "cpu"
    device = torch.device(device_str)
    
    if vb.device == DeviceType.CUDA:
        # Move vb to CPU first to read data
        vb_cpu = vb.clone_to(DeviceType.CPU, 0)
        tensor_cpu = torch.empty(shape, dtype=dtype, device='cpu')
        ptr = tensor_cpu.data_ptr()
        ctypes.memmove(ptr, vb_cpu.data_ptr(), vb_cpu.size_bytes())
        tensor = tensor_cpu.to(device)
    else:
        tensor = torch.empty(shape, dtype=dtype, device='cpu')
        ptr = tensor.data_ptr()
        ctypes.memmove(ptr, vb.data_ptr(), vb.size_bytes())
        
    return tensor
