import sys
import os
import torch

# Try loading C++ core first so child modules can inherit from it
try:
    build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build'))
    sys.path.insert(0, build_dir)
    import visionstream as vs
    HAS_CPP_CORE = True
except ImportError:
    HAS_CPP_CORE = False

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from modules.dataloader.dali_loader import COCODataLoader
from modules.preprocessing.basic_transforms import PreprocessingNode
from modules.external_codec.neural_codec import NeuralCodecNode
from modules.vision_model.yolo_node import YoloInferenceNode
from tools.metrics.evaluator import Evaluator

def run_baseline_pipeline():
    print("=== VisionStream Phase 2 Baseline Evaluation ===")
    
    if HAS_CPP_CORE:
        manager = vs.GraphManager()
    else:
        # Dummy manager if C++ module didn't compile
        class DummyManager:
            def __init__(self): self.nodes = []
            def add_node(self, node): self.nodes.append(node)
            def execute(self, inp):
                for n in self.nodes: inp = n.process(inp)
                return inp
            def get_latencies(self): return {n.name: 1.0 for n in self.nodes}
        manager = DummyManager()
        print("[Warning] C++ core not found. Running in Python-only mock mode.")

    # 1. Initialize Nodes
    prep_node = PreprocessingNode("Preprocessor", target_size=(640, 640))
    codec_node = NeuralCodecNode("Neural_Codec_AE", precision=16)
    yolo_node = YoloInferenceNode("YOLO_Detector", model_name="yolov8n.pt")
    
    manager.add_node(prep_node)
    manager.add_node(codec_node)
    manager.add_node(yolo_node)

    # 2. Setup Evaluator
    evaluator = Evaluator()

    # 3. Dummy Data Loop 
    # (In reality, loop over `COCODataLoader`, here we mock torch tensor for smoke test)
    print("\n[Pipeline] Generating dummy input batch...")
    dummy_input = torch.randint(0, 255, (1, 3, 720, 1280), dtype=torch.uint8, device="cuda:0")

    print("\n[Pipeline] Executing Graph...")
    # For Phase 2 baseline, C++ GraphManager strictly enforces VisionBuffer inputs/outputs.
    # Since YOLO outputs Bounding Boxes (not VisionBuffers), and we use pure PyTorch tensors,
    # we execute the Python pipeline manually over the added nodes.
    results = dummy_input
    for node in [prep_node, codec_node, yolo_node]:
        results = node.process(results)
    
    # 4. Evaluation
    evaluator.add_batch_results(dummy_input, None, None, results)
    mAP = evaluator.compute_map()
    
    print("\n=== Metrics ===")
    print(f"Computed mAP: {mAP}")
    
    try:
        latencies = manager.get_latencies()
        for k, v in latencies.items():
            print(f"Latency [{k}]: {v:.2f} ms")
    except Exception:
        pass
        
    print("Phase 2 Baseline Pipeine test completed successfully.")
    
if __name__ == "__main__":
    run_baseline_pipeline()
