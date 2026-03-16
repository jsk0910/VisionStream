from setuptools import setup, find_packages

setup(
    name="visionstream",
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "pillow",
        "pyyaml",
        "pytorch-msssim",
        "opencv-python",
        "matplotlib",
        "tqdm",
    ],
    extras_require={
        "models": ["timm>=0.9.0", "ultralytics>=8.0.0", "transformers>=4.30.0"],
        "metrics": ["lpips", "pytorch-fid", "pycocotools"],
        "video": ["ffmpeg-python", "decord"],
        "dev": ["streamlit", "pytest", "wandb", "black", "isort"],
    },
    python_requires=">=3.8",
)
