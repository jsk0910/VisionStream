"""
Video Loader Adapter — decord and ffmpeg-python based video readers.
"""
import torch
import numpy as np
from typing import Iterator

try:
    import decord
    from decord import VideoReader, cpu, gpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

try:
    import ffmpeg
    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False

class DecordVideoLoader:
    """High-performance video loader using decord."""
    
    def __init__(self, video_path: str, device: str = "cpu", batch_size: int = 1):
        if not HAS_DECORD:
            raise ImportError("Please install decord: pip install decord")
            
        self.video_path = video_path
        self.batch_size = batch_size
        
        ctx = cpu(0)
        if device.startswith("cuda"):
            try:
                device_id = int(device.split(":")[1]) if ":" in device else 0
                ctx = gpu(device_id)
            except Exception:
                pass # fallback to cpu
                
        self.vr = VideoReader(video_path, ctx=ctx)
        self.num_frames = len(self.vr)
        self.fps = self.vr.get_avg_fps()
        
    def __len__(self) -> int:
        return self.num_frames
        
    def get_batch(self, start_idx: int) -> torch.Tensor:
        """Returns frames [B, C, H, W] in [0, 1] range."""
        end_idx = min(start_idx + self.batch_size, self.num_frames)
        indices = range(start_idx, end_idx)
        
        # decord returns [B, H, W, C] in uint8
        frames = self.vr.get_batch(indices).asnumpy()
        
        # Convert to torch tensor [B, C, H, W] float
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        return tensor

    def __iter__(self) -> Iterator[torch.Tensor]:
        for i in range(0, self.num_frames, self.batch_size):
            yield self.get_batch(i)

class FFmpegVideoLoader:
    """Video loader using ffmpeg-python (useful for wide format support)."""
    
    def __init__(self, video_path: str, batch_size: int = 1):
        if not HAS_FFMPEG:
            raise ImportError("Please install ffmpeg-python: pip install ffmpeg-python")
            
        self.video_path = video_path
        self.batch_size = batch_size
        
        # Probe video to get specs
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        self.width = int(video_info['width'])
        self.height = int(video_info['height'])
        self.num_frames = int(video_info.get('nb_frames', 0))
        
        # Parse framerate (can be fraction like 30000/1001)
        fps_str = video_info.get('avg_frame_rate', '30/1')
        num, den = map(int, fps_str.split('/'))
        self.fps = num / den if den > 0 else 30.0
        
    def __len__(self) -> int:
        return self.num_frames
        
    def __iter__(self) -> Iterator[torch.Tensor]:
        # Stream frames from ffmpeg
        process = (
            ffmpeg
            .input(self.video_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        
        frame_size = self.width * self.height * 3
        batch = []
        
        while True:
            in_bytes = process.stdout.read(frame_size)
            if not in_bytes:
                break
                
            frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
            batch.append(frame)
            
            if len(batch) == self.batch_size:
                tensor = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float() / 255.0
                yield tensor
                batch = []
                
        if batch:
            tensor = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float() / 255.0
            yield tensor
            
        process.stdout.close()
        process.wait()
