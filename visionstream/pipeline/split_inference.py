"""
Split Inference Pipeline
========================
The core pipeline for Video Coding for Machines (VCM) and Split Computing research.

Users configure the four key components:
    1. split_point   — which layer to split the model at
    2. feature_codec — how to compress the extracted feature map (optional)
    3. transmit_fn   — how to transmit the compressed bits (optional)
    4. resume_device — where to run the second half of inference

Example (workspace usage):
    from visionstream.pipeline.split_inference import SplitInferencePipeline
    from visionstream.codecs.intra.jpeg import JPEGCodec
    from my_model import MyDetectionModel  # BaseVisionModel subclass

    pipeline = SplitInferencePipeline(
        model=MyDetectionModel("yolo26n"),
        split_point="backbone.stage3",
        feature_codec=JPEGCodec(quality=50),
        transmit_fn=None,          # None = no transmission simulation
        resume_device="cuda:0",
    )
    result = pipeline.run(image_tensor)
    print(pipeline.get_metrics())  # bpp, encode_ms, decode_ms, etc.
"""

import torch
from typing import Any, Callable, Dict, Optional


class SplitInferencePipeline:
    """
    Configurable Split Inference Pipeline for VCM / Split Computing research.

    The pipeline runs in four stages:
        [Edge]   1. Extract feature map at split_point
        [Codec]  2. Compress feature map with feature_codec (if provided)
        [Net]    3. Transmit compressed bits with transmit_fn (if provided)
        [Server] 4. Decompress and resume inference on resume_device

    All stages are optional except stage 1 (extraction) and 4 (resume).
    If feature_codec is None, raw feature tensors are passed directly.
    If transmit_fn is None, no transmission simulation is performed.

    Args:
        model:         An instance with extract_features() and resume_inference().
                       Should be a SplitVisionModel or compatible wrapper.
        split_point:   Named layer to split at (passed to model.extract_features).
        feature_codec: Optional BaseIntraCodec instance for feature compression.
        transmit_fn:   Optional callable: bytes -> bytes (transmission simulation).
                       Use to model bandwidth constraints, packet loss, etc.
        resume_device: Device string for second-half inference ("cuda:0", "cpu", ...).
    """

    def __init__(
        self,
        model: Any,
        split_point: str,
        feature_codec: Optional[Any] = None,
        transmit_fn: Optional[Callable] = None,
        resume_device: str = "cuda",
    ):
        self.model = model
        self.split_point = split_point
        self.feature_codec = feature_codec
        self.transmit_fn = transmit_fn
        self.resume_device = resume_device
        self._last_metrics: Dict[str, Any] = {}

    def run(self, original_input: torch.Tensor) -> Any:
        """
        Execute the full split inference pipeline.

        Args:
            original_input: Input image tensor [B, C, H, W] in [0, 1].

        Returns:
            Downstream task output (detections, masks, logits, etc.)
        """
        metrics: Dict[str, Any] = {"split_point": self.split_point}

        # ── Stage 1: Edge-side feature extraction ──────────────────────
        feature_map = self.model.extract_features(original_input)
        metrics["feature_shape"] = tuple(feature_map.shape)

        # ── Stage 2: Feature compression (optional) ────────────────────
        if self.feature_codec is not None:
            compressed_payload = self.feature_codec.compress(feature_map)
            metrics.update({
                "bpp":       compressed_payload.get("bpp", 0.0),
                "bytes":     compressed_payload.get("bytes", 0),
                "encode_ms": compressed_payload.get("encode_ms", 0.0),
            })
            bits = compressed_payload.get("bitstream", compressed_payload)
        else:
            bits = feature_map
            metrics["bpp"] = 0.0

        # ── Stage 3: Transmission simulation (optional) ────────────────
        if self.transmit_fn is not None:
            received = self.transmit_fn(bits)
            metrics["transmitted"] = True
        else:
            received = bits
            metrics["transmitted"] = False

        # ── Stage 4: Server-side decompression + inference ─────────────
        if self.feature_codec is not None:
            # Rebuild the payload dict that decompress() expects
            if isinstance(compressed_payload, dict):
                compressed_payload["bitstream"] = received
                feat_decoded = self.feature_codec.decompress(
                    compressed_payload, feature_map.shape
                )
            else:
                feat_decoded = received
            metrics["decode_ms"] = compressed_payload.get("decode_ms", 0.0)
        else:
            feat_decoded = received

        # Move features to resume device
        if hasattr(feat_decoded, "to"):
            feat_decoded = feat_decoded.to(self.resume_device)

        output = self.model.resume_inference(original_input, feat_decoded)
        self._last_metrics = metrics
        return output

    def get_metrics(self) -> Dict[str, Any]:
        """Return metrics from the last pipeline run.

        Keys (when feature_codec is set):
            split_point:   str   — which layer was split
            feature_shape: tuple — shape of the extracted feature map
            bpp:           float — bits per pixel for feature codec
            bytes:         int   — total compressed bytes
            encode_ms:     float — feature encoding latency
            decode_ms:     float — feature decoding latency
            transmitted:   bool  — whether transmit_fn was called
        """
        return self._last_metrics
