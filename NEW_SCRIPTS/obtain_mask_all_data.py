import numpy as np
import tifffile as tiff
from PIL import Image
from pathlib import Path

# ===============================
# Cloud mask
# ===============================
def cloud_mask(img):

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    NIR = img[:, :, 3]

    brightness = (R + G + B) / 3

    bright_thresh = 0.2
    nir_thresh = 0.2

    cloud = (brightness > bright_thresh) & (NIR > nir_thresh)

    mask = np.ones_like(R)
    mask[cloud] = 0

    mask = mask[..., np.newaxis]

    return mask.astype(np.float32)


# ===============================
# Guardar máscara
# ===============================
def save_mask(mask, path):

    mask2d = mask[:, :, 0]
    mask_img = (mask2d * 255).astype(np.uint8)

    Image.fromarray(mask_img).save(path)


# ===============================
# Paths
# ===============================
dataset_root = Path("/mnt/compartida/mbravo_cps/Datasets/CTGAN/CTGAN/Sen2_MTC/dataset/Sen2_MTC")

output_root = Path("/mnt/compartida/mbravo_cps/DOCTORADO/DiffCR_results/data/MASKS_NEW")

output_root.mkdir(parents=True, exist_ok=True)

# ===============================
# Recorrer dataset
# ===============================
tif_files = list(dataset_root.rglob("*.tif"))

print(f"Found {len(tif_files)} images")

for tif_path in tif_files:

    print(f"Processing {tif_path.name}")

    img = tiff.imread(tif_path)
    img = img / 10000.0

    mask = cloud_mask(img)

    # obtener ruta relativa
    relative_path = tif_path.relative_to(dataset_root)

    # obtener región (primer folder)
    region = relative_path.parts[0]

    # carpeta de salida
    output_dir = output_root / region
    output_dir.mkdir(parents=True, exist_ok=True)

    # construir nombre: REGION + índices del nombre original
    stem = tif_path.stem  # ejemplo: T12TUR_R027_0_0 o T12TUR_R027_0
    if stem.startswith(region + "_"):
        indices = stem[len(region) + 1:]
    else:
        parts = stem.split("_", 1)
        indices = parts[1] if len(parts) > 1 else parts[0]

    # Sólo procesar archivos con al menos un '_' en los índices (p.ej. "0_0").
    # Evita generar archivos como REGION_0_mask.png si existen sólo indices simples.
    if "_" not in indices:
        print(f"Skipping single-index file: {tif_path.name}")
        continue

    out_name = f"{region}_{indices}_mask.png"

    save_mask(mask, output_dir / out_name)

