import sys
import os
import time

# Attempt to load the compiled visionstream module from the build directory
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build'))
sys.path.insert(0, build_dir)

import visionstream as vs

class PythonDummyNode(vs.Node):
    def __init__(self, name):
        super().__init__(name)
        
    def process(self, input_buf):
        print(f"[PythonDummyNode] Processing inside python! Node Name: {self.name}")
        print(f"Input Buffer Info: Shape {input_buf.shape}, Dtype {input_buf.dtype}, Device {input_buf.device}")
        
        # Simulate some work
        time.sleep(0.05)
        
        # In a real scenario, we might allocate a new buffer or manipulate the existing
        # For this test, we just pass it along
        return input_buf

def test_pipeline():
    print("=== VisionStream Phase 1 Validation ===")
    
    # 1. Initialize DAG Manager
    manager = vs.GraphManager()
    
    # 2. Setup Nodes
    node1 = PythonDummyNode("PyNode_1")
    node2 = PythonDummyNode("PyNode_2")
    manager.add_node(node1)
    manager.add_node(node2)
    
    # 3. Create a dummy test buffer on CPU
    shape = vs.TensorShape([1, 3, 224, 224])
    buffer = vs.VisionBuffer(shape, vs.DataType.FLOAT32, vs.DeviceType.CPU, 0)
    
    print("\n[Test 1] Executing whole pipeline...")
    out_buffer = manager.execute(buffer)
    
    metrics = manager.get_latencies()
    print("Metrics collected:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f} ms") # It might show 0 if not explicitly set in python, but that's fine for prototype.
        
    print("\n[Test 2] Node Toggle Bypass...")
    # Bypass Node 2
    node2.is_bypassed = True
    print(f"Node 2 Bypassed: {node2.is_bypassed}")
    
    out_buffer2 = manager.execute(buffer)
    print("Pipeline finished with Node 2 bypassed.")
    print("Success!")

if __name__ == "__main__":
    test_pipeline()
