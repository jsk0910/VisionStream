#!/usr/bin/env python3
"""
run_experiment.py — Config-driven VisionStream experiment runner.

Usage:
    python run_experiment.py --config configs/default.yaml
    python run_experiment.py --config configs/v2_elic.yaml
    python run_experiment.py --list               # show all registered components
"""

import argparse
import os
import sys
import json
import time
import yaml
import torch
from datetime import datetime
from typing import Dict, List

# Ensure project root is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "build"))

# Import registry
from modules.registry import (
    get_codec, get_vision_model, get_dataset, get_metric, get_transform,
    list_all,
)

# Trigger auto-discovery of all built-in implementations
import modules.codecs           # noqa: F401
import modules.vision_models    # noqa: F401
import modules.datasets         # noqa: F401
import tools.metrics.builtin_metrics  # noqa: F401
import modules.preprocessing.builtin_transforms  # noqa: F401


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_transforms(transform_configs: List[dict]):
    """Instantiate a list of transforms from config."""
    transforms = []
    for tc in transform_configs:
        name = tc["name"]
        kwargs = {k: v for k, v in tc.items() if k != "name"}
        transforms.append(get_transform(name)(**kwargs))
    return transforms


def apply_transforms(x: torch.Tensor, transforms) -> torch.Tensor:
    for t in transforms:
        x = t(x)
    return x


def run(config: dict):
    pipe = config["pipeline"]
    out_cfg = config.get("output", {})
    verbose = out_cfg.get("verbose", True)
    device = pipe.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Instantiate components from registry ──
    if verbose:
        print("=" * 60)
        print("  VisionStream Experiment Runner")
        print("=" * 60)

    # Dataset
    ds_cls = get_dataset(pipe["dataset"])
    ds_kwargs = pipe.get("dataset_args", {})
    dataset = ds_cls(**ds_kwargs)
    if verbose:
        print(f"[Dataset]  {pipe['dataset']} — {len(dataset)} samples")

    # Transforms
    transforms = build_transforms(pipe.get("transforms", []))
    if verbose:
        print(f"[Transforms] {transforms}")

    # Codec
    codec_cls = get_codec(pipe["codec"])
    codec_kwargs = pipe.get("codec_args", {})
    codec = codec_cls(**codec_kwargs)
    if verbose:
        print(f"[Codec]    {pipe['codec']}")

    # Vision Model (optional)
    vision_model = None
    if pipe.get("vision_model"):
        vm_cls = get_vision_model(pipe["vision_model"])
        vm_kwargs = pipe.get("vision_model_args", {})
        vision_model = vm_cls(**vm_kwargs)
        if verbose:
            print(f"[Vision]   {pipe['vision_model']} ({vision_model.get_task_type()})")

    # Metrics
    metric_names = pipe.get("metrics", ["psnr"])
    metrics = {name: get_metric(name)() for name in metric_names}
    if verbose:
        print(f"[Metrics]  {metric_names}")
        print("-" * 60)

    # ── Run experiment ──
    results = []
    t_start = time.time()

    for idx in range(len(dataset)):
        sample = dataset[idx]
        img = sample["image"]
        fname = sample["filename"]

        # Apply transforms
        img = apply_transforms(img, transforms)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        img = img.to(device)

        # Codec: compress + decompress
        x_hat, codec_info = codec.forward(img)
        x_hat = x_hat.to(device)

        # Compute metrics
        row = {"image": fname}
        for mname, metric in metrics.items():
            val = metric.compute(img, x_hat, bpp=codec_info.get("bpp"))
            row[mname] = round(val, 4)
        row["encode_ms"] = round(codec_info.get("encode_ms", 0), 1)
        row["decode_ms"] = round(codec_info.get("decode_ms", 0), 1)

        # Vision model (optional)
        if vision_model is not None:
            vt0 = time.time()
            _ = vision_model.predict(x_hat)
            row["vision_ms"] = round((time.time() - vt0) * 1000, 1)

        results.append(row)
        if verbose:
            metrics_str = " | ".join(f"{k}={v}" for k, v in row.items() if k != "image")
            print(f"  [{idx+1:3d}/{len(dataset)}] {fname}: {metrics_str}")

    elapsed = time.time() - t_start

    # ── Aggregation ──
    if verbose:
        print("-" * 60)
        print("  AVERAGES:")
    summary = {}
    for mname in metric_names:
        vals = [r[mname] for r in results]
        avg = sum(vals) / len(vals)
        summary[mname] = round(avg, 4)
        if verbose:
            m_obj = metrics[mname]
            print(f"    {m_obj.name():<12} = {avg:.4f}")
    if verbose:
        print(f"    Total time   = {elapsed:.1f}s")
        print("=" * 60)

    # ── Save results ──
    results_dir = out_cfg.get("results_dir", "user_workspace/experiments/")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(results_dir, f"experiment_{pipe['codec']}_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump({"config": config, "summary": summary, "per_image": results}, f, indent=2)
    if verbose:
        print(f"Results saved to {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="VisionStream Experiment Runner")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--list", action="store_true", help="List all registered components")
    args = parser.parse_args()

    if args.list:
        all_reg = list_all()
        print("\n=== VisionStream Registry ===")
        for cat, names in all_reg.items():
            print(f"  [{cat}]: {names if names else '(empty)'}")
        print()
        return

    if not args.config:
        parser.print_help()
        return

    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
