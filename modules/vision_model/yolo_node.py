import torch
try:
    # Requires standard `pip install ultralytics`
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import visionstream as vs
except ImportError:
    class MockNode:
        def __init__(self, name):
            self.name = name
            self.is_bypassed = False
    vs = type('vs', (), {'Node': MockNode})()

class YoloInferenceNode(vs.Node):
    def __init__(self, name, model_name="yolov8n.pt", device="cuda:0"):
        super().__init__(name)
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO is not installed. Run: pip install ultralytics")
            
        print(f"[{self.name}] Loading model {model_name} onto {device}...")
        self.model = YOLO(model_name)
        self.model.to(device)
        self.device = device
        
    def process(self, input_tensor):
        if self.is_bypassed:
            return None
            
        print(f"[{self.name}] Running inference on tensor of shape {input_tensor.shape}...")
        
        # Ultralytics model accepts PyTorch Tensors natively in BCHW format, [0, 1] range.
        # Run inference
        with torch.no_grad():
            # YOLOv8 returns a list of Results objects
            results = self.model(input_tensor, verbose=False)
            
        # Parse standard return format [Boxes, Scores, Classes]
        # Since it's a batch, we return a list of dicts or the raw results object.
        return results
