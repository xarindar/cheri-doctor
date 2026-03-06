"""One-time patch: add figure_id and asset_path to vision figure blocks in document.json."""
import json
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
build_dir = PROJECT_ROOT / "build"
assets_dir = build_dir / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)

print("Loading document.json...")
doc = json.loads((build_dir / "document.json").read_text(encoding="utf-8"))

# Count how many figures need patching
to_patch = sum(
    1 for p in doc["pages"] for b in p["blocks"]
    if b.get("type") == "figure" and not b.get("figure_id")
)
print(f"Found {to_patch} vision figure blocks to patch\n")

patched = 0
exported = 0
for page in doc["pages"]:
    pn = page["page_num"]
    fig_idx = 0
    for block in page["blocks"]:
        if block.get("type") == "figure" and not block.get("figure_id"):
            fig_id = f"fig_p{pn:04d}_{fig_idx:03d}"
            asset_name = f"page_{pn:04d}_fig_{fig_idx:03d}.webp"
            asset_path = assets_dir / asset_name

            if not asset_path.exists():
                page_img_path = build_dir / "pages" / f"page_{pn:04d}.png"
                if page_img_path.exists():
                    img = Image.open(page_img_path)
                    img.save(str(asset_path), "WEBP", quality=85)
                    exported += 1

            block["figure_id"] = fig_id
            block["asset_path"] = str(asset_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            patched += 1

            # Progress bar
            pct = int(patched / to_patch * 100) if to_patch else 100
            bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
            sys.stdout.write(f"\r  [{bar}] {pct:3d}%  ({patched}/{to_patch})  exported: {exported}")
            sys.stdout.flush()

        if block.get("type") == "figure":
            fig_idx += 1

print("\n\nSaving document.json...")
with open(build_dir / "document.json", "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False)

total_figs = sum(1 for p in doc["pages"] for b in p["blocks"] if b.get("type") == "figure")
with_id = sum(1 for p in doc["pages"] for b in p["blocks"] if b.get("type") == "figure" and b.get("figure_id"))
print(f"\nDone! Patched {patched} blocks, exported {exported} WebP images")
print(f"Total figure blocks: {total_figs}, with figure_id: {with_id}")
