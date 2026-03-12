import numpy as np
from PIL import Image
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse
import csv

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--source", required=True)
parser.add_argument("-d", "--dest", required=True)

args = parser.parse_args()

root = Path(args.source)

# carpeta de salida fija
csv_dir = Path("/mnt/compartida/mbravo_cps/DOCTORADO/DiffCR_results/metrics_results")
csv_dir.mkdir(parents=True, exist_ok=True)

csv_path = csv_dir / "metrics_results.csv"

images = list(root.glob("*.png"))

gt_dict = {}
out_dict = {}

# separar GT y Out
for img in images:

    name = img.stem

    if name.startswith("GT_"):
        key = name.replace("GT_", "")
        gt_dict[key] = img

    elif name.startswith("Out_"):
        key = name.replace("Out_", "")
        out_dict[key] = img

keys = sorted(set(gt_dict.keys()) & set(out_dict.keys()))

rows = []

for k in keys:

    gt_path = gt_dict[k]
    pred_path = out_dict[k]

    gt = np.array(Image.open(gt_path)).astype(np.float32)
    pred = np.array(Image.open(pred_path)).astype(np.float32)

    p = psnr(gt, pred, data_range=255)
    s = ssim(gt, pred, channel_axis=2, data_range=255)

    rows.append([
        gt_path.name,
        p,
        s,
        str(gt_path),
        str(pred_path)
    ])

    print(f"{gt_path.name}  PSNR:{p:.3f}  SSIM:{s:.4f}")

# guardar CSV
with open(csv_path, "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["name","psnr","ssim","gt_path","pred_path"])

    for r in rows:
        writer.writerow(r)

print("\n======================")
print("CSV saved in:", csv_path)
print("Images evaluated:", len(rows))
print("Average PSNR:", np.mean([r[1] for r in rows]))
print("Average SSIM:", np.mean([r[2] for r in rows]))