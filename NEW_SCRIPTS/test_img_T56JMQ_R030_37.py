import shutil
from pathlib import Path

# ===============================
# Paths
# ===============================

folder_T56JMQ_R030 = Path("/mnt/compartida/mbravo_cps/Datasets/CTGAN/CTGAN/Sen2_MTC/dataset/Sen2_MTC/T56JMQ_R030")
folder_cloud = folder_T56JMQ_R030 / "cloud"

folder_fused_imgs = Path("/mnt/compartida/mbravo_cps/DOCTORADO/DiffCR_results/data/FUSED_IMAGES_NEW/T56JMQ_R030")

output_folder = Path("/mnt/compartida/mbravo_cps/DOCTORADO/DiffCR_results/data/TEST_img_T56JMQ_R030_37")
output_folder.mkdir(parents=True, exist_ok=True)

# ===============================
# Files que quiero
# ===============================

img_fused = folder_fused_imgs / "T56JMQ_R030_37_fused.tif"
img_1 = folder_cloud / "T56JMQ_R030_37_1.tif"
img_2 = folder_cloud / "T56JMQ_R030_37_2.tif"

# ===============================
# Orden deseado
# ===============================

ordered_files = [
    img_fused,
    img_1,
    img_2
]

# ===============================
# Copiar manteniendo nombre
# ===============================

for i, src in enumerate(ordered_files):
    dst = output_folder / f"{i+1:02d}_{src.name}"
    shutil.copy(src, dst)
    print(f"Copiado: {src} -> {dst}")

print("Proceso terminado.")