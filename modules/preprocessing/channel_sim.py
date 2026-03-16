"""
Channel Simulator Node
Simulates common communication channel degradations such as packet drops and AWGN (Additive White Gaussian Noise)
to test the resilience of the VisionStream pipeline (e.g., VCM).
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Union

class ChannelSimulatorBase(nn.Module):
    def __init__(self, mode: str = "awgn", **kwargs):
        super().__init__()
        self.mode = mode.lower()
        self.params = kwargs

    def forward(self, x: Union[torch.Tensor, bytes, Dict[str, Any]]) -> Union[torch.Tensor, bytes, Dict[str, Any]]:
        """
        Applies channel simulation based on input type.
        """
        if self.mode == "awgn" and isinstance(x, torch.Tensor):
            return self._apply_awgn(x)
        elif self.mode == "packet_drop" and isinstance(x, torch.Tensor):
            return self._apply_tensor_drop(x)
        elif self.mode == "packet_drop" and isinstance(x, bytes):
            return self._apply_byte_drop(x)
        elif self.mode == "packet_drop" and isinstance(x, dict) and "bitstream" in x:
            # Codec dictionary payload
            x_copy = x.copy()
            x_copy["bitstream"] = self._apply_byte_drop(x["bitstream"])
            return x_copy
        else:
            # Fallback (unsupported mode/type combination)
            return x

    def _apply_awgn(self, tensor: torch.Tensor) -> torch.Tensor:
        snr_db = self.params.get("snr_db", 20.0)
        # Assuming signal power is ~1.0 for normalized tensors [0, 1] variance
        # Power of signal / Power of noise = 10^(SNR/10)
        # std_dev = sqrt( P_signal / 10^(SNR/10) )
        signal_power = torch.mean(tensor ** 2)
        noise_variance = signal_power / (10 ** (snr_db / 10.0))
        noise = torch.randn_like(tensor) * torch.sqrt(noise_variance)
        return tensor + noise

    def _apply_tensor_drop(self, tensor: torch.Tensor) -> torch.Tensor:
        """Randomly drops elements (e.g. features) by zeroing them out."""
        drop_rate = self.params.get("drop_rate", 0.1)
        if drop_rate <= 0: return tensor
        mask = torch.rand_like(tensor) > drop_rate
        return tensor * mask.float()

    def _apply_byte_drop(self, data: bytes) -> bytes:
        """
        Simulates payload byte corruption or loss.
        Warning: For standard codecs (JPEG/H.264), altering bytes will likely crash the decoder
        unless error correction/resilience is used. For resilient learned codecs, it tests robustness.
        """
        drop_rate = self.params.get("drop_rate", 0.05)
        if drop_rate <= 0: return data

        data_array = bytearray(data)
        import random
        num_drops = int(len(data_array) * drop_rate)
        # Randomly zero out bytes
        indices = random.sample(range(len(data_array)), num_drops)
        for i in indices:
            data_array[i] = 0
            
        return bytes(data_array)
