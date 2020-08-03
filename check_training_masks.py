import os
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
masks_fns = pd.read_csv('list_classified_masks.txt', sep='\t').iloc[:,0].values

idx = 0
n_pages = int(np.ceil(len(masks_fns) / 6))
with PdfPages('check_masks.pdf') as pdf:
    for page in range(n_pages):
        _, ax = plt.subplots(3, 4, figsize=(20,12))
        ax = ax.ravel()
        for i in range(0,12,2):
            mask_fn = masks_fns[idx]
            img_fn = mask_fn.split('/')[-2].replace('/','.tif')
            img = io.imread(img_fn)
            mask = io.imread(mask_fn)
            mask = mask[:,:,[2,0,1]]
            mask[:,:,0] = 0
            io.imshow(img,ax=ax[i])
            ax[i].set_title(masks_fns[idx][17:])
            io.imshow(mask.astype('float'),ax=ax[i+1])
            ax[i+1].set_title(masks_fns[idx][17:])
            idx += 1
            print('Prossessed {}'.format(mask_fn))
            if idx == len(masks_fns) -1:
                break
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
