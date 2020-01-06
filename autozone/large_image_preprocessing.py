import skimage as ski
import numpy as np
from czifile import imread
import os

def czi_roi2tiff(czifilename, dapi_col=2):
    """Extract each roi from raw czi image and save them as individual files
    """
    img = imread(czifilename)
    roi_dims = img.shape[-6]
    for roi in range(roi_dims):
        dapi = img[0, 0, roi, 0, dapi_col, :, :, 0]
        crit = dapi == 0
        valid_cols = np.sum(crit, axis=0) != crit.shape[0]
        valid_rows = np.sum(crit, axis=1) != crit.shape[1]
        first_row, first_col = np.argmax(valid_rows), np.argmax(valid_cols)
        last_row = np.min(
            (
                np.arange(crit.shape[0], 0, -1)[np.argmax(valid_rows[::-1])] + 1,
                crit.shape[0],
            )
        )
        last_col = np.min(
            (
                np.arange(crit.shape[1], 0, -1)[np.argmax(valid_cols[::-1])] + 1,
                crit.shape[1],
            )
        )
        roi_img = img[0, 0, roi, 0, :, first_row:last_row, first_col:last_col, 0]
        roi_img = np.swapaxes(roi_img, 0, 2)[:, :, ::-1]
        roi_img_fn = " ".join([czifilename.replace(".czi", ""), "roi", str(roi),'.tiff'])
        ski.io.imsave(roi_img_fn, roi_img)

os.chdir('T:/images/2020-1-1-Big tif')
czi_roi2tiff('2019_11_26__1069-AAV-1-100-F245.czi')