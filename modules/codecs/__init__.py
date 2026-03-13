"""modules/codecs/__init__.py — Auto-import all codec implementations."""
from pathlib import Path
import importlib

# Auto-discover and import all .py files in this package
_pkg_dir = Path(__file__).parent
for _f in _pkg_dir.glob("*.py"):
    if _f.name.startswith("_"):
        continue
    importlib.import_module(f".{_f.stem}", package=__name__)
