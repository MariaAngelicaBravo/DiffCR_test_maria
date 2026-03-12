import os
import numpy as np
import tifffile as tiff
from PIL import Image
from pathlib import Path

# ===============================
# Paths
# ===============================

input_root  = Path("/mnt/compartida/mbravo_cps/DOCTORADO/DiffCR_results/data/NEW_DATASET")
output_root = Path("/mnt/compartida/mbravo_cps/DOCTORADO/DiffCR_results/data/NEW_DATASET_PNG")
output_root.mkdir(parents=True, exist_ok=True)

def convert_tif_to_png(tif_path, png_path):

    img = tiff.imread(tif_path).astype(np.float32)

    # usar RGB
    rgb = img[:, :, :3]

    rgb = np.clip(rgb, 0, 2000)

    rgb = rgb - np.nanmin(rgb)

    max_val = np.nanmax(rgb)
    if max_val == 0:
        rgb = 255 * np.ones_like(rgb)
    else:
        rgb = 255 * (rgb / max_val)

    rgb = np.nan_to_num(rgb)
    rgb = rgb.astype(np.uint8)

    Image.fromarray(rgb).save(png_path)


# ===============================
# Recorrer dataset
# ===============================

for tile in input_root.iterdir():

    if not tile.is_dir():
        continue

    for typ in tile.iterdir():

        if not typ.is_dir():
            continue

        save_dir = output_root / tile.name / typ.name
        save_dir.mkdir(parents=True, exist_ok=True)

        for tif_file in typ.glob("*.tif"):

            out_file = save_dir / (tif_file.stem + ".png")

            convert_tif_to_png(tif_file, out_file)

            print("Saved:", out_file)

print("All images converted to PNG!")