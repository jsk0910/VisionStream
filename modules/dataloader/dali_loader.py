import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import torch
import sys
import os

# To map to VisionBuffer later if needed
# import visionstream as vs

class COCODataLoader:
    def __init__(self, data_dir, annotations_file, batch_size=1, num_threads=4, device_id=0):
        self.data_dir = data_dir
        self.annotations_file = annotations_file
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.device_id = device_id
        
        self.pipeline = self._build_pipeline()
        self.pipeline.build()
        
    def _build_pipeline(self):
        @dali.pipeline_def(batch_size=self.batch_size, num_threads=self.num_threads, device_id=self.device_id)
        def coco_pipeline():
            inputs, bboxes, labels, polygons, vertices = fn.readers.coco(
                file_root=self.data_dir,
                annotations_file=self.annotations_file,
                polygon_masks=True,
                ratio=True,
                name="Reader"
            )
            
            # Decode image on GPU directly
            images = fn.decoders.image(inputs, device="mixed", output_type=types.RGB)
            
            return images, bboxes, labels
            
        return coco_pipeline()

    def get_next_batch(self):
        """
        Runs the DALI pipeline and returns output.
        Returns:
            images (DALI TensorListGPU): The decoded standard RGB images.
            bboxes (DALI TensorListCPU): Bounding boxes for objects.
            labels (DALI TensorListCPU): Labels for the boxes.
        """
        pipe_out = self.pipeline.run()
        images, bboxes, labels = pipe_out
        return images, bboxes, labels

    def to_pytorch(self, dali_tensor):
        """
        Helper method to convert DALI GPU Tensor to PyTorch GPU Tensor without host copy.
        """
        import torch.utils.dlpack
        # DALI has dlpack support
        dlcapsule = dali_tensor.as_tensor().to_dlpack()
        torch_tensor = torch.utils.dlpack.from_dlpack(dlcapsule)
        return torch_tensor

# Quick smoke test if run directly
if __name__ == "__main__":
    print("DALI Loader module initialized.")
    # loader = COCODataLoader("/path/to/coco/train2017", "/path/to/coco/annotations/instances_train2017.json")
    # imgs, boxes, labels = loader.get_next_batch()
