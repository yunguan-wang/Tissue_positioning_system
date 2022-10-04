import sys
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tps.find_zones import *
from tps.plotting import *
from tps.segmentation import *
import pandas as pd

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
# input_fn = '/endosome/archive/shared/zhu_wang/TPS_paper/TIF-20211214/list_classified_masks.txt'
input_fn = sys.argv[1]
masks_fns = sorted(
    pd.read_csv(input_fn, sep='\t', header=None).iloc[:,0].dropna().values)
print(len(masks_fns))
idx = 0
n_pages = int(np.ceil(len(masks_fns) / 3))
with PdfPages('check_masks.pdf') as pdf:
    for page in range(n_pages):
        _, ax = plt.subplots(3, 2, figsize=(20,12))
        ax = ax.ravel()
        for i in range(0,6,2):
            if idx == len(masks_fns):
                break
            mask_fn = masks_fns[idx]
            title = mask_fn.split('/')[-2]
            img_fn = mask_fn.replace('/cv_pv_masks.tif','.tif')
            img = io.imread(img_fn)
            mask = io.imread(mask_fn)
            io.imshow(img,ax=ax[i])
            ax[i].set_title(title)
            io.imshow(mask,ax=ax[i+1])
            ax[i+1].set_title(title)
            idx += 1
            print('Prossessed {}'.format(mask_fn))
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
