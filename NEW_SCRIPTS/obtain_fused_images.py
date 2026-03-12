import numpy as np
import tifffile as tiff
from pathlib import Path
from PIL import Image

dataset_root = Path("/mnt/compartida/mbravo_cps/Datasets/CTGAN/CTGAN/Sen2_MTC/dataset/Sen2_MTC")
masks_root   = Path("/mnt/compartida/mbravo_cps/DOCTORADO/DiffCR_results/data/MASKS_NEW")
output_root  = Path("/mnt/compartida/mbravo_cps/DOCTORADO/DiffCR_results/data/FUSED_IMAGES_NEW")
output_root.mkdir(parents=True, exist_ok=True)

def fuse_images(img_list, mask_list):
    I1, I2, I3 = img_list
    M1, M2, M3 = mask_list
    fused = I1 * M1 + I2 * (1 - M1) * M2 + I3 * (1 - M1) * (1 - M2) * M3
    return fused.astype(np.float32)

regions = [d for d in dataset_root.iterdir() if d.is_dir()]

for region_path in regions:
    region_name = region_path.name
    print(f"Processing region: {region_name}")

    # obtener todas las imágenes y máscaras
    img_files = sorted(region_path.rglob("*.tif"))
    mask_files = sorted((masks_root / region_name).glob("*_mask.png"))

    # agrupar por tile base name
    tiles = {}
    for f in img_files:
        # quitar la parte del tiempo: ej. T12TUR_R027_0_0.tif -> T12TUR_R027_0
        tile_base = "_".join(f.stem.split("_")[:-1])
        tiles.setdefault(tile_base, []).append(f)

    for tile_base, imgs in tiles.items():
        if len(imgs) != 3:
            print(f"Skipping {tile_base}: expected 3 times, got {len(imgs)}")
            continue

        # cargar imágenes en orden de tiempo
        imgs_sorted = sorted(imgs, key=lambda x: x.stem)
        imgs_array = [tiff.imread(f)/10000.0 for f in imgs_sorted]

        # cargar máscaras correspondientes
        masks_array = []
        for f in imgs_sorted:
            mask_path = masks_root / region_name / (f.stem + "_mask.png")
            mask = np.array(Image.open(mask_path)).astype(np.float32) / 255.0
            mask = mask[..., np.newaxis]
            masks_array.append(mask)

        # fusión
        fused = fuse_images(imgs_array, masks_array)

        # guardar
        output_dir = output_root / region_name
        output_dir.mkdir(parents=True, exist_ok=True)
        fused_name = tile_base + "_fused.tif"
        tiff.imwrite(output_dir / fused_name, fused)

        print(f"Saved fused image: {output_dir / fused_name}")

print("All regions processed!")