from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parent.parent / "build"
PAUSE_FILE = BUILD_DIR / "PAUSE"

if not BUILD_DIR.exists():
    BUILD_DIR.mkdir(parents=True)

with open(PAUSE_FILE, "w") as f:
    f.write("paused")

print(f"✅ Paused! Created {PAUSE_FILE}")
print("The pipeline will pause at the next page boundary.")
