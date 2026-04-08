"""
FFmpeg Codec Adapter — H.264/H.265 (HEVC) wrappers via ffmpeg-python.
This wrapper relies on external ffmpeg binaries for encoding/decoding,
avoiding direct implementation of patented standard codecs.
"""
import os
import time
import tempfile
import torch
import numpy as np
from typing import Any, Dict, Tuple
from modules.registry import BaseCodec, register_codec

try:
    import ffmpeg
    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False

class FFmpegVideoCodec(BaseCodec):
    """Wrapper for ffmpeg-python to encode/decode video tensors."""
    
    def __init__(self, codec_name: str = "libx264", crf: int = 23, preset: str = "medium", **kwargs):
        if not HAS_FFMPEG:
            raise ImportError("Please install ffmpeg-python: pip install ffmpeg-python")
            
        self.codec_name = codec_name
        self.crf = crf
        self.preset = preset
        
    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Compress a sequence of frames.
        Args:
            x: [B, C, H, W] float tensor in [0, 1]. Here B acts as sequence length.
        Returns:
            Dict containing bitstream, bps (bits per pixel), and encoding latency.
        """
        B, C, H, W = x.shape
        # Convert tensor [B, C, H, W] in [0, 1] to numpy [B, H, W, C] in uint8
        frames_np = (x.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        
        start_time = time.time()
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Write frames to pipe and encode
            process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{W}x{H}', framerate=30)
                .output(tmp_path, vcodec=self.codec_name, crf=self.crf, preset=self.preset, pix_fmt='yuv420p', loglevel="error")
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
            
            for frame in frames_np:
                process.stdin.write(frame.tobytes())
                
            process.stdin.close()
            process.wait()
            
            encode_ms = (time.time() - start_time) * 1000.0
            
            # Read back the bitstream
            with open(tmp_path, 'rb') as f:
                bitstream = f.read()
                
            num_pixels = B * H * W
            total_bits = len(bitstream) * 8
            bpp = total_bits / num_pixels if num_pixels > 0 else 0
            
            return {
                "bitstream": bitstream,
                "bpp": bpp,
                "encode_ms": encode_ms,
                "shape": (B, C, H, W),
                "fps": 30
            }
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    def decompress(self, payload: Dict[str, Any], shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Decompress a bitstream back to frames.
        """
        bitstream = payload["bitstream"]
        B, C, H, W = payload.get("shape", shape)
        
        start_time = time.time()
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(bitstream)
            tmp_path = tmp_file.name
            
        try:
            out, _ = (
                ffmpeg
                .input(tmp_path)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            decode_ms = (time.time() - start_time) * 1000.0
            payload["decode_ms"] = decode_ms
            
            if not out:
                return torch.zeros(payload.get("shape", shape))
                
            frames_np = np.frombuffer(out, np.uint8).reshape([-1, H, W, 3])
            
            # Match required length if possible
            if len(frames_np) > B:
                frames_np = frames_np[:B]
                
            # Convert back to tensor [B, C, H, W]
            x_hat = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
            
            # Pad if we decoded fewer frames than expected (should be rare)
            if x_hat.shape[0] < B:
                pad_size = B - x_hat.shape[0]
                pad = torch.zeros((pad_size, C, H, W))
                x_hat = torch.cat([x_hat, pad], dim=0)
                
            return x_hat
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


@register_codec("h264")
class H264Codec(FFmpegVideoCodec):
    def __init__(self, crf: int = 23, preset: str = "medium", **kwargs):
        # Fallback to software libx264 if NVENC x264 is not preferred or available,  
        # but in a complete setup `h264_nvenc` could be used.
        super().__init__(codec_name="libx264", crf=crf, preset=preset, **kwargs)


@register_codec("h265")
class H265Codec(FFmpegVideoCodec):
    def __init__(self, crf: int = 23, preset: str = "medium", **kwargs):
        super().__init__(codec_name="libx265", crf=crf, preset=preset, **kwargs)
