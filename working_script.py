import pandas as pd
import skimage as ski
from skimage import io
import os
from segmentation import *
from plotting import *
from find_zones import *

tert1 = "T:/images/New Tif/New Tif/Normal  30%/Tert-20190628-5TAM-m1-9.tif"
tert2 = "input/tert-1.tif"
img_fn = tert1
img = io.imread(img_fn)

masks = segmenting_vessels_gs_assisted(img, vessel_size_t=2)
cv_features = extract_features(masks, img[:, :, 1], q1=0.75, q2=0.99, step=0.1)
cv_labels, pv_labels = pv_classifier(cv_features.loc[:, "I0":], masks)

cv_masks = masks * np.isin(masks, cv_labels).copy()
pv_masks = masks * np.isin(masks, pv_labels).copy()
masks = shrink_cv_masks(cv_masks, pv_masks, img[:, :, 2])

coords_pixel, coords_cv, coords_pv = find_pv_cv_coords(masks, cv_labels, pv_labels)
orphans = find_orphans(masks, cv_labels, pv_labels)
zone_crit = get_distance_projection(
    coords_pixel, coords_cv, coords_pv, cosine_filter=True
)
zone_crit[masks != 0] = -1
zone_crit[(zone_crit > 1) | (zone_crit < 0)] = -1
zone_crit[orphans] = -1

zones = create_zones(masks, zone_crit, cv_labels, pv_labels, num_zones=9)
zone_int = plot_zone_int_probs(
    img[:, :, 0], img[:, :, 2], zones, dapi_cutoff="otsu", plot_type="probs"
)

# plotting scribbles

# plot_zones = zones.copy()
# plot_zones[plot_zones == -1] = 7
# plot_zones[plot_zones == 255] = 6
# plot_zones = plot_zones * 20

# tar_int = img[:, :, 0].copy()
# int_cutoff = ski.filters.threshold_otsu(tar_int)

# plt.imshow((tar_int > int_cutoff) & (zones == -1))
# plt.imshow(plot_zones, alpha=0.5)

# tar_int[int_signal_mask & (zones == -1)] = 0

# zone_int = plot_zone_int_probs(
#     tar_int, img[:, :, 2], zones, dapi_cutoff="otsu", plot_type="ppz"
# )

# int_cutoff = ski.filters.threshold_otsu(tar_int)
# int_signal_mask = tar_int > int_cutoff
# dapi_int = img[:, :, 0]
# dapi_cutoff = ski.filters.threshold_otsu(dapi_int)

# total_pos_int = (zones != 0) & int_signal_mask & (dapi_int > dapi_cutoff)

# plt.imshow(total_pos_int)
# plt.imshow(plot_zones, alpha=0.5)
