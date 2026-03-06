from pathlib import Path
import os

BUILD_DIR = Path(__file__).resolve().parent.parent / "build"
PAUSE_FILE = BUILD_DIR / "PAUSE"

if PAUSE_FILE.exists():
    os.remove(PAUSE_FILE)
    print(f"✅ Resumed! Removed {PAUSE_FILE}")
    print("The pipeline should continue shortly.")
else:
    print(f"⚠️  No pause file found at {PAUSE_FILE}")
