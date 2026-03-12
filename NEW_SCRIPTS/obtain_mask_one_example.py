import numpy as np
import tifffile as tiff
from PIL import Image

# generar máscaras de nubes a partir de las imágenes originales usando un método simple basado en el brillo y el canal NIR.
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

    # agregar dimensión de canal
    mask = mask[..., np.newaxis]

    print(f"Cloud mask computed with shape: {mask.shape}") # mask.shape = (256, 256, 1)

    return mask

# guardar máscara como imagen PNG para visualización
def save_mask(mask, name):

    # quitar dimensión extra
    mask2d = mask[:,:,0]

    # convertir a 0-255 para visualizar
    mask_img = (mask2d * 255).astype(np.uint8)

    Image.fromarray(mask_img).save(name)


img1 = tiff.imread("/mnt/compartida/mbravo_cps/Datasets/CTGAN/CTGAN/Sen2_MTC/dataset/Sen2_MTC/T12TUR_R027/cloud/T12TUR_R027_0_0.tif")
img2 = tiff.imread("/mnt/compartida/mbravo_cps/Datasets/CTGAN/CTGAN/Sen2_MTC/dataset/Sen2_MTC/T12TUR_R027/cloud/T12TUR_R027_0_1.tif")
img3 = tiff.imread("/mnt/compartida/mbravo_cps/Datasets/CTGAN/CTGAN/Sen2_MTC/dataset/Sen2_MTC/T12TUR_R027/cloud/T12TUR_R027_0_2.tif")

img1 = img1 / 10000.0
img2 = img2 / 10000.0
img3 = img3 / 10000.0

print(img1.min(), img1.max())
print(img2.min(), img2.max())
print(img3.min(), img3.max())

mask1 = cloud_mask(img1)
mask2 = cloud_mask(img2)
mask3 = cloud_mask(img3)

save_mask(mask1, "/mnt/Home-Group/mbravo_cps/DOCTORADO/DiffCR_paper_me/NEW_SCRIPTS/RESULTS_NEW/mask1.png")
save_mask(mask2, "/mnt/Home-Group/mbravo_cps/DOCTORADO/DiffCR_paper_me/NEW_SCRIPTS/RESULTS_NEW/mask2.png")
save_mask(mask3, "/mnt/Home-Group/mbravo_cps/DOCTORADO/DiffCR_paper_me/NEW_SCRIPTS/RESULTS_NEW/mask3.png")
