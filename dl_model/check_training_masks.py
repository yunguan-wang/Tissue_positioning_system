import os
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
MASKS_PATH = "/endosome/work/InternalMedicine/s190548/TPS/tps_results"

for cond in ['2w','48w', 'DDC_control', 'DDC']:
    folders = os.listdir(os.path.join(MASKS_PATH, cond))
    mask_fns = []
    img_fns = []
    for fd in folders:
        mask_fns.append(os.path.join(MASKS_PATH, cond, fd, 'refined_masks.tif'))
        img_fns.append(os.path.join(MASKS_PATH, cond, fd, 'raw_image_scaled.tif'))
    idx = 0
    mask_fns = sorted(mask_fns)
    img_fns = sorted(img_fns)
    n_pages = int(np.ceil(len(mask_fns) / 6))
    with PdfPages(cond + '_check_masks.pdf') as pdf:
        for page in range(n_pages):
            _, ax = plt.subplots(3, 4, figsize=(20,12))
            ax = ax.ravel()
            for i in range(0,12,2):
                mask_fn = mask_fns[idx]
                img_fn = img_fns[idx]
                img = io.imread(img_fn)
                mask = io.imread(mask_fn)
                # mask = mask[:,:,[2,0,1]]
                # mask[:,:,0] = 0
                io.imshow(img,ax=ax[i])
                ax[i].set_title(img_fn.split('/')[-2])
                io.imshow(mask,ax=ax[i+1])
                ax[i+1].set_title(mask_fn.split('/')[-2])
                idx += 1
                if idx == len(mask_fns) -1:
                    break
            if idx == len(mask_fns) -1:
                break
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
