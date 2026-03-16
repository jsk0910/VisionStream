"""
VisionStream Interactive Dashboard — Phase 8 (Enhanced)
A Streamlit-based web UI that provides:
  1. Single-image Neural Codec analysis with YOLO overlay
  2. JPEG baseline comparison (side-by-side)
  3. Batch Kodak dataset benchmark tab
  4. Pixel-level difference heatmap
  5. Architecture / project status overview
"""

import streamlit as st
import os
import sys
import io as sysio
import glob
import time
import torch
import numpy as np
import cv2
from PIL import Image
from pytorch_msssim import ms_ssim

# ──────────────────────────── paths ────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "build"))
sys.path.insert(0, os.path.join(ROOT, "user_workspace/custom_codecs/learned_compression"))

try:
    import visionstream as vs
    from modules.vision_model.yolo_node import YoloInferenceNode
    from model_v2 import HybridCompressionModelV2
    HAS_BACKEND = True
except ImportError as exc:
    HAS_BACKEND = False
    _IMPORT_ERR = str(exc)

# ──────────────────────────── cached loaders ───────────────────
@st.cache_resource
def load_v2_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = HybridCompressionModelV2(device=device).to(device)
    model.eval()
    ckpt = os.path.join(ROOT, "user_workspace/custom_codecs/learned_compression/checkpoint_v2.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    return model, device

@st.cache_resource
def load_yolo(device):
    return YoloInferenceNode("yolo", model_name="yolov8n.pt", device=device)

@st.cache_resource
def build_cdf():
    total = 1 << 16
    cdf = [0]
    for i in range(255):
        pm = max(1, int(total * (0.5 ** (i + 1))))
        cdf.append(min(total, cdf[-1] + pm))
    cdf[-1] = total
    for i in range(len(cdf) - 1):
        if cdf[i + 1] <= cdf[i]:
            cdf[i + 1] = cdf[i] + 1
    if cdf[-1] > total:
        cdf = [int((v / cdf[-1]) * total) for v in cdf]
        cdf[-1] = total
    return cdf

# ──────────────────────────── helpers ──────────────────────────
def _pad(x, p=64):
    h, w = x.shape[2], x.shape[3]
    hp = (p - h % p) % p
    wp = (p - w % p) % p
    return torch.nn.functional.pad(x, (0, wp, 0, hp)), hp, wp

def _to_pil(t):
    return Image.fromarray((t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

def _compress_jpeg(pil_img, quality):
    buf = sysio.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    size_bytes = buf.tell()
    buf.seek(0)
    return Image.open(buf).convert("RGB"), size_bytes

def _run_v2_codec(img_tensor, model, device, cdf):
    """Run V2 neural codec and return (recon_tensor, metrics_dict)."""
    img_tensor = img_tensor.to(device)
    padded, _, _ = _pad(img_tensor)
    oh, ow = img_tensor.shape[2], img_tensor.shape[3]

    with torch.no_grad():
        t0 = time.time()
        y = model.encoder(padded * 255.0)
        y_hat = model.quantize(y, is_training=False)
        z = model.hyper_encoder(y)
        z_hat = model.quantize(z, is_training=False)

        y_sym = torch.clamp(y_hat + 128, 0, 255).to(torch.uint8).flatten().cpu().tolist()
        z_sym = torch.clamp(z_hat + 128, 0, 255).to(torch.uint8).flatten().cpu().tolist()
        syms = y_sym + z_sym
        idxs = [0] * len(syms)

        tae = time.time()
        bs = vs.ArithmeticCoder.encode(syms, idxs, cdf, [256], [0], 16)
        ae_ms = (time.time() - tae) * 1000

        bpp = (len(bs) * 8) / (oh * ow)

        tad = time.time()
        _ = vs.ArithmeticCoder.decode(bs, idxs, cdf, [256], [0], 16)
        ad_ms = (time.time() - tad) * 1000

        x_hat = model.decoder(y_hat) / 255.0
        recon = torch.clamp(x_hat[:, :, :oh, :ow], 0.0, 1.0)

        mse = torch.nn.functional.mse_loss(recon, img_tensor)
        psnr = (10 * torch.log10(1.0 / mse)).item() if mse > 0 else 100.0
        ssim = ms_ssim(recon, img_tensor, data_range=1.0, size_average=True).item()

        total_ms = (time.time() - t0) * 1000

    return recon, {
        "bpp": bpp, "psnr": psnr, "msssim": ssim,
        "ae_ms": ae_ms, "ad_ms": ad_ms, "total_ms": total_ms,
        "bytes": len(bs),
    }

def _draw_yolo(img_np, results):
    canvas = img_np.copy()
    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls)
        conf = float(box.conf)
        label = f"{results[0].names[cls]} {conf:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return canvas, len(boxes)

def _diff_heatmap(orig_pil, recon_pil):
    a = np.array(orig_pil).astype(np.float32)
    b = np.array(recon_pil).astype(np.float32)
    # Resize b to match a if shapes differ
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    diff = np.mean(np.abs(a - b), axis=2)
    diff_norm = (diff / max(diff.max(), 1e-5) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# ──────────────────────────── UI ───────────────────────────────
st.set_page_config(page_title="VisionStream Dashboard", layout="wide", page_icon="🔬")

st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    h1 {font-size: 2rem !important;}
    .stMetric {background: #1e1e2e; border-radius: 12px; padding: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("🔬 VisionStream — Hybrid Vision Research Dashboard")

if not HAS_BACKEND:
    st.error(f"Backend unavailable: {_IMPORT_ERR}")
    st.stop()

# ── Tabs ──
tab_single, tab_batch, tab_mlops, tab_sim, tab_arch = st.tabs([
    "🖼️ Single Image Analysis",
    "📊 Batch Kodak Benchmark",
    "📈 MLOps Experiments",
    "📶 Channel Sim Demo",
    "🏗️ Architecture Overview",
])

# ═════════════════════════ TAB 1: Single Image ═════════════════
with tab_single:
    sidebar, main = st.columns([1, 3])

    with sidebar:
        st.subheader("⚙️ Settings")
        kodak_dir = os.path.join(ROOT, "data/kodak")
        kodak_files = sorted(glob.glob(os.path.join(kodak_dir, "*.png")))

        src = st.radio("Image Source", ("Kodak Dataset", "Upload"), horizontal=True)
        pil_img = None
        if src == "Kodak Dataset" and kodak_files:
            sel = st.selectbox("Image", [os.path.basename(f) for f in kodak_files])
            pil_img = Image.open(os.path.join(kodak_dir, sel)).convert("RGB")
        elif src == "Upload":
            up = st.file_uploader("Upload", type=["png", "jpg", "jpeg"])
            if up:
                pil_img = Image.open(up).convert("RGB")

        jpeg_q = st.slider("JPEG Quality (for comparison)", 1, 100, 50)
        run_yolo = st.checkbox("Run YOLO Detection", value=True)

        if pil_img:
            st.image(pil_img, caption="Original", use_column_width=True)

        run = st.button("🚀 Execute Pipeline", use_container_width=True)

    with main:
        if run and pil_img:
            model, device = load_v2_model()
            cdf = build_cdf()

            arr = np.array(pil_img)
            img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            # ── V2 Codec ──
            with st.spinner("Running V2 Neural Codec + C++ Entropy…"):
                recon_t, m = _run_v2_codec(img_t, model, device, cdf)
            recon_pil = _to_pil(recon_t)

            # ── JPEG Baseline ──
            jpeg_pil, jpeg_bytes = _compress_jpeg(pil_img, jpeg_q)
            jpeg_bpp = (jpeg_bytes * 8) / (arr.shape[0] * arr.shape[1])
            jpeg_t = torch.from_numpy(np.array(jpeg_pil)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            jpeg_mse = torch.nn.functional.mse_loss(jpeg_t, img_t)
            jpeg_psnr = (10 * torch.log10(1.0 / jpeg_mse)).item() if jpeg_mse > 0 else 100.0

            # ── Comparison images ──
            st.markdown("### 🔄 Visual Comparison")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(pil_img, caption="Original", use_column_width=True)
            with c2:
                st.image(recon_pil, caption="V2 Neural Codec", use_column_width=True)
            with c3:
                st.image(jpeg_pil, caption=f"JPEG Q={jpeg_q}", use_column_width=True)

            # ── Metrics table ──
            st.markdown("### 📊 Metrics Comparison")
            mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
            mc1.metric("V2 PSNR", f"{m['psnr']:.2f} dB")
            mc2.metric("V2 MS-SSIM", f"{m['msssim']:.4f}")
            mc3.metric("V2 BPP", f"{m['bpp']:.4f}")
            mc4.metric("JPEG PSNR", f"{jpeg_psnr:.2f} dB")
            mc5.metric("JPEG BPP", f"{jpeg_bpp:.4f}")
            mc6.metric("V2 Bitstream", f"{m['bytes'] / 1024:.1f} KB")

            # ── Latency ──
            st.markdown("### ⏱️ Latency")
            l1, l2, l3 = st.columns(3)
            l1.metric("C++ AE Encode", f"{m['ae_ms']:.1f} ms")
            l2.metric("C++ AE Decode", f"{m['ad_ms']:.1f} ms")
            l3.metric("Total Codec", f"{m['total_ms']:.1f} ms")

            # ── Diff Heatmap ──
            st.markdown("### 🌡️ Pixel Difference Heatmap")
            hm1, hm2 = st.columns(2)
            with hm1:
                st.image(_diff_heatmap(pil_img, recon_pil), caption="V2 Codec Diff (brighter = more error)", use_column_width=True)
            with hm2:
                st.image(_diff_heatmap(pil_img, jpeg_pil), caption=f"JPEG Q={jpeg_q} Diff", use_column_width=True)

            # ── YOLO ──
            if run_yolo:
                st.markdown("### 🎯 YOLO Object Detection on Reconstructed Image")
                with st.spinner("Running YOLOv8…"):
                    yolo_node = load_yolo(device)
                    ty0 = time.time()
                    yolo_res = yolo_node.process(recon_t)
                    yolo_ms = (time.time() - ty0) * 1000

                recon_np = np.array(recon_pil)
                drawn, n_obj = _draw_yolo(recon_np, yolo_res)
                st.image(drawn, caption=f"YOLO Detections ({n_obj} objects) — {yolo_ms:.0f} ms", use_column_width=True)

            st.success(f"✅ Pipeline completed in {m['total_ms'] + (yolo_ms if run_yolo else 0):.0f} ms total.")
        elif not run:
            st.info("👈 Select an image and click **Execute Pipeline** to start.")

# ═════════════════════════ TAB 2: Batch Benchmark ══════════════
with tab_batch:
    st.subheader("📊 Batch Kodak Benchmark (24 Images)")
    st.markdown("Run the V2 Neural Codec against all Kodak images and display aggregate metrics.")

    if st.button("▶️ Run Full Kodak Benchmark", key="batch_btn"):
        model, device = load_v2_model()
        cdf = build_cdf()
        kodak_dir = os.path.join(ROOT, "data/kodak")
        kodak_files = sorted(glob.glob(os.path.join(kodak_dir, "*.png")))

        if not kodak_files:
            st.error("Kodak dataset not found.")
        else:
            rows = []
            progress = st.progress(0, text="Processing…")
            for idx, fpath in enumerate(kodak_files):
                fname = os.path.basename(fpath)
                pil = Image.open(fpath).convert("RGB")
                arr = np.array(pil)
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0

                _, m = _run_v2_codec(t, model, device, cdf)

                # JPEG Q50 baseline
                _, jbytes = _compress_jpeg(pil, 50)
                jbpp = (jbytes * 8) / (arr.shape[0] * arr.shape[1])

                rows.append({
                    "Image": fname,
                    "V2 PSNR (dB)": round(m["psnr"], 2),
                    "V2 MS-SSIM": round(m["msssim"], 4),
                    "V2 BPP": round(m["bpp"], 4),
                    "JPEG Q50 BPP": round(jbpp, 4),
                    "AE Encode (ms)": round(m["ae_ms"], 1),
                    "AE Decode (ms)": round(m["ad_ms"], 1),
                })
                progress.progress((idx + 1) / len(kodak_files), text=f"Processing {fname}…")

            progress.empty()

            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            st.markdown("#### Averages")
            ac1, ac2, ac3, ac4 = st.columns(4)
            ac1.metric("Avg PSNR", f"{df['V2 PSNR (dB)'].mean():.2f} dB")
            ac2.metric("Avg MS-SSIM", f"{df['V2 MS-SSIM'].mean():.4f}")
            ac3.metric("Avg V2 BPP", f"{df['V2 BPP'].mean():.4f}")
            ac4.metric("Avg JPEG Q50 BPP", f"{df['JPEG Q50 BPP'].mean():.4f}")
    else:
        st.info("Click the button above to start the batch benchmark.")

# ═════════════════════════ TAB 3: MLOps Experiments ════════════
with tab_mlops:
    st.subheader("📈 MLOps Experiment Log")
    st.markdown("Automated tracking of runs via `versioning.py` and `BaseLogger`.")
    
    def load_experiments(exp_dir: str = "user_workspace/experiments"):
        import json
        import pandas as pd
        records = []
        if os.path.isdir(os.path.join(ROOT, exp_dir)):
            for run_id in os.listdir(os.path.join(ROOT, exp_dir)):
                cfg_path = os.path.join(ROOT, exp_dir, run_id, "config.json")
                if os.path.exists(cfg_path):
                    with open(cfg_path, 'r') as f:
                        cfg = json.load(f)
                        records.append({
                            "Run ID": cfg.get("run_id"),
                            "Timestamp": cfg.get("timestamp"),
                            "Git Hash": cfg.get("git_hash"),
                            "BPP": cfg.get("config", {}).get("bpp", None),
                            "Metric": cfg.get("config", {}).get("metric_name", "Unknown"),
                            "Score": cfg.get("config", {}).get("metric_score", None)
                        })
        if not records:
            return pd.DataFrame(columns=["Run ID", "Timestamp", "Git Hash", "BPP", "Metric", "Score"])
        df = pd.DataFrame(records)
        df = df.sort_values("Timestamp", ascending=False)
        return df
        
    df_exp = load_experiments()
    st.dataframe(df_exp, use_container_width=True)

# ═════════════════════════ TAB 4: Channel Sim Demo ═════════════
with tab_sim:
    st.subheader("📶 Channel Simulator Demo")
    st.markdown("Visualize the effect of AWGN and Packet drops on features or frames.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("AWGN Noise (SNR dB)", 0, 40, 20)
    with col2:
        st.slider("Packet Drop Rate", 0.0, 1.0, 0.1)
        
    st.info("Interactive nodes are wired in `vcm_pipeline.py`. To execute this demo, launch the pipeline runner.")

# ═════════════════════════ TAB 5: Architecture ═════════════════
with tab_arch:
    st.subheader("🏗️ VisionStream Architecture Overview")
    st.markdown("""
**VisionStream** is a Hybrid Vision Research Framework blending C++/CUDA high-performance backends
with PyTorch-based neural compression and vision models.

---

#### Project Phases (Status)

| Phase | Description | Status |
|:---:|:---|:---:|
| **1** | Core C++ Engine (`VisionBuffer`, `GraphManager`, `pybind11`) | ✅ Complete |
| **2** | Baseline System (DALI, Preprocessing, YOLO, Evaluator) | ✅ Complete |
| **3** | C++/CUDA Arithmetic Coder + Neural Codec Python Node | ✅ Complete |
| **4** | Kodak Dataset Integration & End-to-End BPP/Latency Test | ✅ Complete |
| **5** | Custom Learned Compression (Hyper-Prior, GDN, R-D Loss) | ✅ Complete |
| **6** | Next-Gen V2 Codec (ELIC + GMM + Attention) — **+1.82 dB** | ✅ Complete |
| **7** | E2E DAG Pipeline (Preprocessor → Codec → YOLO) | ✅ Complete |
| **8** | Extended Metrics (MS-SSIM) & Interactive Dashboard | 🔄 In Progress |

---

#### Key Modules

| Module | Language | Path |
|:---|:---:|:---|
| `VisionBuffer` | C++ / CUDA | `core/memory/vision_buffer.h` |
| `GraphManager` / `Node` | C++ | `core/graph/graph_manager.h` |
| `ArithmeticCoder` | C++ / CUDA | `core/codec/arithmetic_coder.h` |
| `HybridCompressionModelV2` | Python (PyTorch) | `user_workspace/.../model_v2.py` |
| `EncoderV2` / `DecoderV2` | Python (PyTorch) | `user_workspace/.../network_v2.py` |
| `ELICContextModel` / `RateDistortionLossV2` | Python (PyTorch) | `user_workspace/.../entropy_gmm.py` |
| `YoloInferenceNode` | Python (Ultralytics) | `modules/vision_model/yolo_node.py` |
| `PreprocessingNode` | Python | `modules/preprocessing/basic_transforms.py` |

---

#### V2 Neural Codec Architecture
```
Input Image (3×H×W)
    │
    ▼
┌─────────────────┐
│  EncoderV2      │  ResBlocks + GDN + CBAM Attention
│  (stride=2 ×4)  │  → 192×(H/16)×(W/16)
└────────┬────────┘
         │  y
    ┌────┴────┐
    │ Quantize│  (Uniform noise / Round)
    └────┬────┘
         │  ŷ ──────────────────────┐
    ┌────┴──────┐             ┌─────┴──────┐
    │HyperEnc V2│             │ELICContext  │
    │→ z → ẑ    │──→ HyperDec │  Model     │
    └───────────┘   → ψ       │ [16,16,32, │
                     │         │  128] grps │
                     └────┬────┘
                          │  GMM params (K=3)
                          │  weights, means, scales
                          ▼
                   ┌──────────────┐
                   │ C++ Arith.   │
                   │ Encoder →    │──→ Bitstream (bytes)
                   │ Decoder      │
                   └──────────────┘
         │  ŷ
    ┌────┴────────┐
    │  DecoderV2  │  CBAM + ResBlocks + IGDN
    │  (upsample) │  → 3×H×W
    └─────────────┘
         │
         ▼
    Reconstructed Image → YOLOv8 Detection
```
""")
