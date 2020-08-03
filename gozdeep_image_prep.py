import pandas as pd
from skimage import io, morphology
import numpy as np
import os
from goz.large_image_processing import find_valid_crops
from goz.segmentation import find_boundry

input_imgs=[
    '/home2/s190548/work_zhu/images/20200621-AAV/1wk/AAV-2.tif',
    '/home2/s190548/work_zhu/images/20200621-AAV/1wk/AAV-1.tif',
    '/home2/s190548/work_zhu/images/20200621-AAV/1wk/AAV-3.tif',
    ]
x=4000
y=2000
for img_fn in input_imgs:
    img = io.imread(img_fn)
    # b_masks = find_boundry(img[:,:,2])
    output_prefix = (
        '/home2/s190548/work_zhu/gozdeep_input/')
    try:
        os.makedirs(output_prefix)
    except:
        pass
    valid_crops = find_valid_crops(img[:, :, 2],x, y,0)
    crop_fn = img_fn.split('/')[-1].strip('.tif')
    for i,crop in enumerate(valid_crops):
        img_crop = img[crop[0]:crop[1],crop[2]:crop[3]]
        crop_fn_prefix = '_'.join([str(x) for x in crop])
        io.imsave(
            os.path.join(output_prefix,'_'.join([crop_fn,crop_fn_prefix,'.tif'])),
            img_crop)
