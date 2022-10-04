#%%
from skimage import io
import os
import numpy as np
from tps.large_image_processing import find_valid_crops
from tps.segmentation import find_boundry

input_path = '/endosome/archive/shared/zhu_wang/TPS_paper/TIF-20211214/'
input_imgs=['Mup3-L5.tif']
input_imgs = [input_path + '/' + x for x in input_imgs]
output_path = '/endosome/work/InternalMedicine/s190548/TPS/tps_results/bigtif/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
#%%
x = 4000
y = 2000
dapi_channel = 2
for img_fn in input_imgs:
    img = io.imread(img_fn)
    img = img[1000:9000, 0:12000]
    b_masks = find_boundry(img[:,:,2])
    bgnd_median = np.median(img[:,:,2][~b_masks])
    norm_dapi = img[:,:,2].astype(int) - bgnd_median
    norm_dapi[norm_dapi<0] = 0
    norm_dapi = 255*norm_dapi/norm_dapi.max()
    img[:,:,2] = norm_dapi.astype('uint8')
    valid_crops = find_valid_crops(img[:, :, dapi_channel],x, y,0,valid_only=False)
    crop_fn = img_fn.split('/')[-1].strip('.tif')
    for i,crop in enumerate(valid_crops):
        img_crop = img[crop[0]:crop[1],crop[2]:crop[3]]
        crop_fn_prefix = '_'.join([str(int(x)) for x in crop])
        io.imsave(
            os.path.join(output_path,'_'.join([crop_fn,crop_fn_prefix]) + '.tif'),
            img_crop)

# %%
