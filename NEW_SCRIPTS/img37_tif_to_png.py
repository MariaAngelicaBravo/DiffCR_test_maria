import numpy as np
import tifffile as tiff
from PIL import Image
from pathlib import Path

# ===============================
# Paths
# ===============================

tif_folder = Path("/mnt/compartida/mbravo_cps/DOCTORADO/DiffCR_results/data/TEST_img_T56JMQ_R030_37")

output_folder = Path("/mnt/compartida/mbravo_cps/DOCTORADO/DiffCR_results/data/TEST_img_T56JMQ_R030_37_PNG")
output_folder.mkdir(parents=True, exist_ok=True)

# ===============================
# Convert function
# ===============================

def convert_tif_to_png(tif_path, png_path):

    img = tiff.imread(tif_path).astype(np.float32)

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
# Convert all tif
# ===============================

for tif_file in tif_folder.glob("*.tif"):

    out_file = output_folder / (tif_file.stem + ".png")

    convert_tif_to_png(tif_file, out_file)

    print("Saved:", out_file)

print("All images converted!")