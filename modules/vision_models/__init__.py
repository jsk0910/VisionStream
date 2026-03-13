"""modules/vision_models/__init__.py — Auto-import all vision model implementations."""
from pathlib import Path
import importlib

_pkg_dir = Path(__file__).parent
for _f in _pkg_dir.glob("*.py"):
    if _f.name.startswith("_"):
        continue
    importlib.import_module(f".{_f.stem}", package=__name__)
