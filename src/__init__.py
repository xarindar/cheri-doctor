# Metro Manual — Chat With The Manual
# Adds project root and legacy/ to sys.path so legacy modules are importable.
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
_LEGACY_DIR = str(Path(__file__).resolve().parent.parent / "legacy")

for p in (_PROJECT_ROOT, _LEGACY_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
