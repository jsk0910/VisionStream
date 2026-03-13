<p align="center">
  <h1 align="center">🎥 VisionStream</h1>
  <p align="center">
    <b>하이브리드 비전 연구 플랫폼</b> — 압축 코덱 · 비전 태스크를 자유롭게 교체 · 조합 · 비교하는<br>
    C++/CUDA + Python 기반 연구 프레임워크
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-11.8%20|%2012.1%20|%2012.8-green" alt="CUDA">
  <img src="https://img.shields.io/badge/C%2B%2B-17-orange" alt="C++17">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

---

## 📋 목차

- [개요](#-개요)
- [주요 기능](#-주요-기능)
- [아키텍처](#-아키텍처)
- [프로젝트 구조](#-프로젝트-구조)
- [시스템 요구 사항](#-시스템-요구-사항)
- [설치 방법](#-설치-방법)
- [C++ 코어 빌드](#-c-코어-빌드)
- [실행 방법](#-실행-방법)
- [YAML 설정 작성법](#-yaml-설정-작성법)
- [커스텀 컴포넌트 등록](#-커스텀-컴포넌트-등록)
- [벤치마크](#-벤치마크)
- [테스트](#-테스트)
- [Docker](#-docker)
- [로드맵](#-로드맵)

---

## 🔍 개요

**VisionStream**은 이미지/비디오 압축 코덱과 다운스트림 비전 태스크(검출, 분류, 분할, 초해상도, 깊이 추정 등)를 **하나의 DAG 파이프라인**으로 연결하여, 방법론(코덱, 모델, 지표, 전처리)을 자유롭게 교체·조합·비교할 수 있는 연구 플랫폼입니다.

### 설계 철학

> **바퀴를 다시 만들지 않는다.** 표준 백본·체크포인트는 기존 생태계(PyPI + Hub)를 통해 필요할 때만 내려받고, VisionStream이 직접 제공하는 것은 **코어 엔진 + 어댑터 레이어 + 실험 프레임워크**에 집중합니다.

---

## ✨ 주요 기능

| 카테고리 | 기능 |
|:---|:---|
| **C++/CUDA 코어** | ArithmeticCoder (Python 대비 ~80× 처리량), VisionBuffer (Zero-copy GPU 전달), DAG 스케줄러 |
| **압축 코덱** | JPEG, WebP (Pillow 래퍼) + 자체 V2 ELIC 신경망 코덱 (ELIC + GMM + Attention) |
| **비전 모델** | YOLOv8 검출, timm 분류 (ResNet50, ViT, EfficientNet 등), SegFormer 분할, Real-ESRGAN 초해상도, Depth-Anything 깊이 추정 |
| **데이터셋** | Kodak, ImageFolder (빌트인) + COCO, DIV2K 자동 다운로드 |
| **지표** | PSNR, MS-SSIM, BPP (빌트인) + LPIPS, FID, mAP 어댑터 |
| **플러그인 시스템** | `@register_*` 데코레이터 기반 — 코어 코드 수정 없이 새 컴포넌트 추가 |
| **실험 러너** | YAML 기반 설정으로 파이프라인 정의 → 자동 실행 → JSON 결과 저장 |
| **벤치마크** | 모듈 단위 / 백본 / 비전 모델 / 코덱 R-D 커브 / E2E 파이프라인 벤치마크 |
| **멀티 환경** | CUDA 11.8, 12.1, 12.8 × Conda, pip, Docker 지원 |

---

## 🏗 아키텍처

VisionStream은 3-Layer 아키텍처로 구성됩니다:

```
┌──────────────────────────────────────────────────────────────┐
│           Layer 3: Ecosystem Layer  (외부 패키지)              │
│   torchvision · timm · ultralytics · transformers · lpips    │
│   HuggingFace Hub · PyTorch Hub (체크포인트 on-demand)        │
└──────────────────────────────────────────────────────────────┘
          ▲   pip install + 자동 다운로드
          │
┌──────────────────────────────────────────────────────────────┐
│           Layer 2: VisionStream Adapter Layer  (소스 제공)     │
│   modules/registry.py   — ABC + @register_* 데코레이터        │
│   modules/codecs/       — JPEG/WebP 래퍼, V2 ELIC (자체 구현)  │
│   modules/vision_models/— timm · ultralytics 씬 래퍼          │
│   modules/datasets/     — BaseDataset + 다운로드 헬퍼          │
│   tools/metrics/        — BaseMetric + PSNR/MS-SSIM 래퍼      │
└──────────────────────────────────────────────────────────────┘
          ▲   C++/Python 바인딩 (pybind11)
          │
┌──────────────────────────────────────────────────────────────┐
│           Layer 1: VisionStream Core  (C++/CUDA 빌드 필요)    │
│   core/codec/   — ArithmeticCoder (C++/CUDA)                 │
│   core/graph/   — GraphManager · Node · DAG 스케줄러           │
│   core/memory/  — VisionBuffer (Zero-copy CPU/GPU)            │
│   python_api/   — pybind11 바인딩                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 프로젝트 구조

```
VisionStream/
│
├── core/                         # C++/CUDA 코어 (소스 + 빌드)
│   ├── codec/                    #   ArithmeticCoder
│   ├── graph/                    #   GraphManager, Node, DAG
│   └── memory/                   #   VisionBuffer
│
├── modules/                      # Python 어댑터 레이어
│   ├── registry.py               #   ABC + @register_* 시스템
│   ├── codecs/                   #   jpeg_codec.py, v2_elic.py
│   ├── vision_models/            #   yolo, classification, segmentation, sr, depth
│   ├── datasets/                 #   builtin_datasets.py, hub_datasets.py
│   └── preprocessing/            #   torchvision transforms 래퍼
│
├── tools/
│   ├── metrics/                  #   PSNR, MS-SSIM, BPP + lpips/FID 어댑터
│   └── ui/                       #   Streamlit 대시보드
│
├── user_workspace/               # 연구자 샌드박스
│   ├── custom_codecs/            #   자체 코덱 (V1, V2 ELIC 포함)
│   ├── custom_models/            #   파인튜닝 모델
│   └── experiments/              #   실험 결과 (JSON 자동 저장)
│
├── configs/                      # YAML 실험 설정
│   ├── default.yaml              #   JPEG 기준선 파이프라인
│   └── v2_elic.yaml              #   V2 신경망 코덱 파이프라인
│
├── envs/                         # 멀티 환경 설정
│   ├── conda/                    #   environment_cuda{118,121,128}.yml
│   ├── requirements/             #   requirements_cuda{118,121,128}.txt
│   └── docker/                   #   Dockerfile.cuda{118,121,128}
│
├── scripts/                      # 유틸리티 스크립트
├── tests/                        # 단위 · 통합 테스트
├── python_api/                   # pybind11 C++/Python 바인딩
├── docker/                       # Docker 설정
│
├── CMakeLists.txt                # C++ 코어 빌드 설정
├── pyproject.toml                # 계층화된 Python 의존성
├── run_experiment.py             # YAML 기반 실험 진입점
└── run_benchmark.py              # 벤치마크 CLI
```

---

## 💻 시스템 요구 사항

| 항목 | 최소 요구 사항 |
|:---|:---|
| **OS** | Linux (Ubuntu 18.04+) |
| **Python** | 3.8+ |
| **CUDA** | 11.8 / 12.1 / 12.8 |
| **GPU** | NVIDIA GPU (Compute Capability 7.0+) |
| **CMake** | 3.18+ |
| **C++ 컴파일러** | GCC 9+ 또는 Clang 10+ (C++17 지원) |
| **RAM** | 16GB+ 권장 |
| **VRAM** | 8GB+ 권장 |

---

## 🚀 설치 방법

### Option A: Conda (권장)

로컬 개발 및 멀티 프로젝트 환경에 적합합니다.

```bash
# 1. 저장소 클론
git clone https://github.com/your-org/VisionStream.git
cd VisionStream

# 2. Conda 환경 생성 (CUDA 버전에 맞게 선택)
conda env create -f envs/conda/environment_cuda121.yml   # CUDA 12.1
# conda env create -f envs/conda/environment_cuda118.yml # CUDA 11.8
# conda env create -f envs/conda/environment_cuda128.yml # CUDA 12.8

# 3. 환경 활성화
conda activate visionstream

# 4. Python 패키지 설치
pip install -e .                     # 코어 의존성만
pip install -e ".[models]"           # + 비전 모델 (timm, ultralytics, transformers)
pip install -e ".[models,metrics]"   # + 지표 (lpips, pycocotools)
pip install -e ".[dev]"              # + 개발 도구 (pytest, wandb, streamlit)
```

### Option B: pip + venv

단일 프로젝트 환경 또는 CI/CD에 적합합니다.

```bash
# 1. 저장소 클론
git clone https://github.com/your-org/VisionStream.git
cd VisionStream

# 2. 가상 환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate

# 3. 의존성 설치 (CUDA 버전에 맞게 선택)
pip install -r envs/requirements/requirements_cuda121.txt

# 4. Python 패키지 설치
pip install -e ".[models,metrics]"
```

### Option C: Docker

완벽한 재현성이 필요한 환경(팀 협업, 논문 재현, 클러스터 배포)에 적합합니다.

```bash
# CUDA 버전 선택 후 빌드 & 실행
cd docker
docker compose up --build
```

### 의존성 계층 구조

`pyproject.toml`에 정의된 계층화된 의존성으로 필요한 패키지만 선택적으로 설치할 수 있습니다:

| 설치 명령 | 포함 패키지 | 사용 사례 |
|:---|:---|:---|
| `pip install -e .` | torch, numpy, pillow, pyyaml, pytorch-msssim, opencv, matplotlib, tqdm | 코덱 연구 (최소 의존성) |
| `pip install -e ".[models]"` | + timm, ultralytics, transformers | 비전 태스크 활용 |
| `pip install -e ".[metrics]"` | + lpips, pytorch-fid, pycocotools | 고급 지표 사용 |
| `pip install -e ".[video]"` | + ffmpeg-python, decord | 비디오 파이프라인 |
| `pip install -e ".[dev]"` | + streamlit, pytest, wandb, black, isort | 개발 · 대시보드 |

---

## 🔨 C++ 코어 빌드

C++/CUDA 코어(ArithmeticCoder, VisionBuffer, GraphManager)를 사용하려면 별도 빌드가 필요합니다:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89"
make -j$(nproc)
cd ..
```

빌드 완료 후 `build/` 디렉토리에 `visionstream` Python 모듈이 생성됩니다.

> **참고**: pybind11은 CMake가 자동으로 가져옵니다 (시스템에 없으면 FetchContent로 다운로드).

---

## ▶️ 실행 방법

### 1. 실험 실행 (YAML 기반 파이프라인)

YAML 설정 파일로 데이터셋, 전처리, 코덱, 비전 모델, 지표를 정의하고 파이프라인을 실행합니다.

```bash
# JPEG 코덱 + YOLOv8 검출 파이프라인 (기본 설정)
python run_experiment.py --config configs/default.yaml

# V2 ELIC 신경망 코덱 + YOLOv8 검출 파이프라인
python run_experiment.py --config configs/v2_elic.yaml
```

결과는 `user_workspace/experiments/` 디렉토리에 JSON 형식으로 자동 저장됩니다.

### 2. 등록된 컴포넌트 확인

현재 사용 가능한 모든 컴포넌트(코덱, 모델, 데이터셋, 지표, 전처리)를 확인합니다.

```bash
python run_experiment.py --list
```

출력 예시:

```
=== VisionStream Registry ===
  [codec]: ['jpeg', 'v2_elic']
  [vision_model]: ['yolov8n', 'resnet50', 'vit_base', 'segformer', 'real_esrgan', 'depth_anything']
  [dataset]: ['kodak', 'image_folder', 'coco', 'div2k']
  [metric]: ['psnr', 'msssim', 'bpp']
  [transform]: ['normalize', 'to_bchw', 'resize', 'center_crop']
```

### 3. 벤치마크 실행

```bash
# 백본 단독 벤치마크 (배치 32, 224×224)
python run_benchmark.py --backbone resnet50

# 비전 모델 벤치마크
python run_benchmark.py --task segmentation --model segformer

# 코덱 R-D 커브 측정
python run_benchmark.py --codec v2_elic

# E2E 파이프라인 벤치마크 (코덱 → 비전 모델)
python run_benchmark.py --e2e --codec jpeg --model yolov8n
```

### 4. Streamlit 대시보드

실험 결과를 시각적으로 탐색할 수 있는 웹 대시보드를 실행합니다.

```bash
bash scripts/launch_dashboard.sh
# 또는
streamlit run tools/ui/app.py
```

---

## 📝 YAML 설정 작성법

`configs/` 디렉토리에 YAML 파일을 작성하여 실험 파이프라인을 정의합니다.

```yaml
# configs/my_experiment.yaml
pipeline:
  dataset: kodak                     # @register_dataset으로 등록된 이름
  dataset_args:
    root: data/kodak

  transforms:                        # 전처리 (순서대로 적용)
    - name: normalize
    - name: to_bchw

  codec: jpeg                        # @register_codec으로 등록된 이름
  codec_args:
    quality: 75

  vision_model: yolov8n              # @register_vision_model으로 등록된 이름
  vision_model_args:
    device: cuda:0
    conf_threshold: 0.25

  metrics:                           # @register_metric으로 등록된 이름 리스트
    - psnr
    - msssim
    - bpp

  device: cuda:0

training:
  epochs: 100
  batch_size: 4
  lr: 0.0001
  lambda_rd: 0.01

output:
  results_dir: user_workspace/experiments/
  save_images: false
  verbose: true
```

---

## 🔌 커스텀 컴포넌트 등록

VisionStream의 플러그인 시스템을 사용하면 코어 코드를 수정하지 않고도 새 컴포넌트를 추가할 수 있습니다.

### 커스텀 코덱

```python
from modules.registry import BaseCodec, register_codec

@register_codec("my_codec")
class MyCodec(BaseCodec):
    def compress(self, x):
        # 압축 로직 구현
        return {"bitstream": ..., "bpp": ..., "encode_ms": ...}

    def decompress(self, payload, shape):
        # 복원 로직 구현
        return reconstructed_tensor
```

### 커스텀 비전 모델

```python
from modules.registry import BaseVisionModel, register_vision_model

@register_vision_model("my_detector")
class MyDetector(BaseVisionModel):
    def predict(self, x):
        # 추론 로직 구현
        return results

    def get_task_type(self):
        return "detection"
```

### 커스텀 데이터셋

```python
from modules.registry import BaseDataset, register_dataset

@register_dataset("my_dataset")
class MyDataset(BaseDataset):
    def __getitem__(self, index):
        return {"image": tensor, "filename": name}

    def __len__(self):
        return self.size
```

### 커스텀 지표

```python
from modules.registry import BaseMetric, register_metric

@register_metric("my_metric")
class MyMetric(BaseMetric):
    def compute(self, original, reconstructed, **kwargs):
        return score

    def name(self):
        return "My Metric"
```

등록 후 YAML 설정의 해당 필드에 등록 이름을 사용하면 바로 파이프라인에서 활용 가능합니다.

---

## 📊 벤치마크

`run_benchmark.py`를 통해 다양한 수준의 벤치마크를 수행할 수 있습니다.

| 벤치마크 유형 | 명령어 | 설명 |
|:---|:---|:---|
| **모듈 단위** | `--module arithmetic_coder` | C++ 코어 모듈 성능 측정 |
| **백본** | `--backbone resnet50` | timm 백본 FPS/VRAM 측정 (batch 32) |
| **비전 모델** | `--task segmentation --model segformer` | 비전 태스크 모델 벤치마크 |
| **코덱 R-D** | `--codec v2_elic` | Rate-Distortion 커브 생성 |
| **E2E** | `--e2e --codec jpeg --model yolov8n` | 코덱→모델 전체 파이프라인 |

결과는 `benchmark_results/` 디렉토리에 **JSON** + **Markdown 리포트**로 자동 저장됩니다.

---

## 🧪 테스트

```bash
# 전체 테스트 실행
pytest tests/ -v

# 개별 테스트 실행
pytest tests/test_pipeline.py -v        # 파이프라인 테스트
pytest tests/test_pipeline_e2e.py -v    # E2E 통합 테스트
pytest tests/test_custom_codec.py -v    # 커스텀 코덱 테스트
pytest tests/test_kodak.py -v           # Kodak 데이터셋 테스트
```

---

## 🐳 Docker

`docker/` 디렉토리에서 Docker 기반 환경을 제공합니다:

```bash
cd docker

# 빌드 및 실행
docker compose up --build

# 또는 직접 빌드
docker build -t visionstream .
docker run --gpus all -it visionstream
```

CUDA 버전별 Dockerfile은 `envs/docker/`에서도 제공됩니다:

```bash
# CUDA 12.1 전용 이미지
docker build -f envs/docker/Dockerfile.cuda121 -t visionstream:cuda121 .
```

---

## 🗺 로드맵

| Phase | 상태 | 내용 |
|:---:|:---:|:---|
| 1–9 | ✅ 완료 | C++ 코어, DAG 파이프라인, V1/V2 신경망 코덱, 레지스트리, 대시보드, 실험 러너 |
| 10 | 🔄 진행 중 | 의존성 구조 정비 + 어댑터 확충 (timm/SegFormer/SR/Depth 래퍼) |
| 11 | 📋 예정 | 성능 벤치마크 자동화 |
| 12 | 📋 예정 | 비디오 파이프라인 (H.264/H.265, 비디오 추적, VMAF) |
| 13 | 📋 예정 | VCM — Feature Map 압축 (Split Model, Task-Aware Loss) |
| 14 | 📋 예정 | MLOps + 풀스택 웹 앱 (Streamlit 대체) |

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
