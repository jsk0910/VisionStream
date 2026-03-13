"""
Benchmark Utilities for VisionStream.
"""
import torch
import time
from typing import Callable, Dict

def measure_cuda_latency(func: Callable, *args, warmups: int = 10, iterations: int = 100, **kwargs) -> float:
    """
    Measures the average execution time of a function on CUDA with proper synchronization.
    Returns: Average latency in milliseconds.
    """
    # Warmup
    for _ in range(warmups):
        func(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    for _ in range(iterations):
        func(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    return ((end_time - start_time) / iterations) * 1000.0

def measure_throughput(func: Callable, *args, data_size: int, iterations: int = 100, **kwargs) -> float:
    """
    Measures throughput (items/sec or bytes/sec).
    Returns: Throughput per second.
    """
    latency_ms = measure_cuda_latency(func, *args, iterations=iterations, **kwargs)
    if latency_ms == 0:
        return 0
    return (data_size / (latency_ms / 1000.0))

def get_gpu_memory_usage() -> Dict[str, float]:
    """
    Returns current GPU memory usage in MB.
    """
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0}
    
    return {
        "allocated": torch.cuda.memory_allocated() / (1024 ** 2),
        "reserved": torch.cuda.memory_reserved() / (1024 ** 2)
    }
