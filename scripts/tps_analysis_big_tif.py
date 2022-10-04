#%%
# import argparse
from tps.find_zones import *
from tps.plotting import *
from tps.segmentation import *
import os
from skimage import io, measure, morphology
import pandas as pd
from scipy.ndimage.morphology import binary_fill_holes
import numpy as np
#%%
def prune_masks(masks, vessel_r_cutoff = 15):
    # iteratively merging masks until it cannot be merged.
    masks = morphology.opening(masks, morphology.disk(5))
    masks = measure.label(masks)
    rpt = pd.DataFrame(
        measure.regionprops_table(
            masks,properties=['label','equivalent_diameter']))
    valid_labels = rpt[rpt.equivalent_diameter>=vessel_r_cutoff].label.values
    masks[~np.isin(masks,valid_labels)] = 0
    pruned_masks = binary_fill_holes(masks>0)
    return pruned_masks.astype(int)

def pool_masks(l_cv_masks, l_pv_masks):
    # merge two non-overlapping lebeled masks together.
    # ensure there is no overlap
    assert(set(np.unique(l_cv_masks))&set(np.unique(l_pv_masks)) == {0})
    p_masks = (l_cv_masks>0) | (l_pv_masks>0)
    p_masks_exp = morphology.binary_dilation(p_masks, morphology.disk(10))
    p_masks_exp = binary_fill_holes(p_masks_exp)
    p_masks_exp = measure.label(p_masks_exp)
    output_masks = np.zeros(p_masks.shape, int)
    cv_labels = []
    pv_labels = []
    for label, region in enumerate(measure.regionprops(p_masks_exp)):
        if region.label == 0:
            continue
        x0,y0,x1,y1 = region.bbox
        _image = region.image
        cv_score = (l_cv_masks[x0:x1,y0:y1][_image]>0).sum()
        pv_score = (l_pv_masks[x0:x1,y0:y1][_image]>0).sum()
        if cv_score > pv_score:
            cv_labels.append(label+1)
        elif cv_score < pv_score:
            pv_labels.append(label+1)
        else:
            continue
        output_masks[x0:x1,y0:y1][_image] = label+1
    cv_masks = output_masks*np.isin(output_masks, cv_labels)
    pv_masks = output_masks*np.isin(output_masks, pv_labels)
    return output_masks, cv_labels, pv_labels, cv_masks, pv_masks

def pool_crops(
    crops, output_size = (30000,30000,3),x0_idx = 2, size_factor=2,
    crop_type={'img':'img', 'masks':'masks'}['img']):
    output = np.zeros(output_size, 'uint8')
    x_max = 0
    y_max = 0
    for crop in crops:
        img_crop = io.imread(crop)
        if crop_type == 'masks':
            for i in range(1,3):
                _l_masks = measure.label(img_crop[:,:,i])
                _pruned_mask = prune_masks(_l_masks)
                img_crop[:,:,i] = _pruned_mask
        coords = crop.split('/')[-1].replace('.tif','').split('_')
        x0,x1,y0,y1 = [
            int(int(x)/size_factor) for x in coords[x0_idx:x0_idx+4]]
        if ((x1-x0) != img_crop.shape[0]) or ((y1-y0) != img_crop.shape[1]):
            output[x0:x1,y0:y1,:] = img_crop[:x1-x0,:y1-y0,:]
        else:
            output[x0:x1,y0:y1,:] = img_crop
        x_max = max(x_max,x1)
        y_max = max(y_max,y1)
    return output[:x_max,:y_max,:]
#%%
# TPS cell size
input_path = '/endosome/work/InternalMedicine/s190548/TPS/tps_results/bigtif/input/'
output_path = '/endosome/work/InternalMedicine/s190548/TPS/tps_results/bigtif/output/'
os.chdir(input_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
#%%
masks_fn = [x for x in os.listdir() if ('mask' in x) and ('tif' in x)]
tifs_fn = [x for x in os.listdir() if ('.tif' in x) & (x not in masks_fn)]
img_fns = ['Mup3']

#%%
for img_fn in img_fns:
    img_tifs = [x for x in tifs_fn if img_fn in x]
    img_masks = [x for x in masks_fn if img_fn in x]
    img = pool_crops(img_tifs,x0_idx=1)
    masks = pool_crops(img_masks, x0_idx=1, crop_type='masks')
    b_mask = find_boundry(img[:,:,2], dapi_t=50)
    img = img * b_mask[:,:,None]
    masks = masks * b_mask[:,:,None]
    cv_masks = masks[:,:,1]>0
    pv_masks = masks[:,:,2]>0
    cv_masks = measure.label(prune_masks(cv_masks))
    pv_masks = measure.label(prune_masks(pv_masks))
    pv_masks[pv_masks>0] = pv_masks[pv_masks>0] + cv_masks.max()
    masks, cv_labels, pv_labels, cv_masks, pv_masks = pool_masks(
        cv_masks, pv_masks)
    _ = plt.figure(figsize=(16,9))
    plot3channels(
    img[:,:,2],
    cv_masks != 0,
    pv_masks != 0,
    fig_name= output_path + '/masks')

    # plot raw image
    _ = plt.figure(figsize=(16,9))
    io.imshow(img)
    plt.savefig(output_path + '/Original_img.pdf')
    plt.close()

    zone_crit = calculate_zone_crit(cv_masks, pv_masks, tolerance=100)
    # Calculate zones
    zones = create_zones(
        masks,
        zone_crit,
        cv_labels,
        pv_labels,
        zone_break_type="equal_quantile",
        num_zones=9,
    )*b_mask
    # Plot zones with image
    plot_zones_only(zones, fig_prefix=output_path + "/zones only")
    io.imsave(output_path + '/rescaled.tif', img)
    io.imsave(output_path + '/rescaled_zones.tif', zones)
