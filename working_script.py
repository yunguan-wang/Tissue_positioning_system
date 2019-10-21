# import pandas as pd
import skimage as ski
from skimage import io
import os
from segmentation import *
from plotting import *
from find_zones import *

liver1 = "input/gls2-1.tif"
liver2 = "input/tert-1.tif"
img_fn = liver2
img = io.imread(img_fn)
img = np.array(img, dtype=np.uint8)
masks = segmenting_vessels_gs_assisted(img, vessel_size_t=2)
cv_features = extract_features(masks, img[:, :, 1], q1=0.75, q2=0.99, step=0.1)
cv_labels, pv_labels = pv_classifier(cv_features.loc[:, "I0":], masks)

# output_mask_fn = "output/" + img_fn.split("/")[1].replace("-1.tif", "_masks.tif")
# output_prefix = img_fn.split("/")[1].replace("-1.tif", "")
# if os.path.exists(output_mask_fn):
#     masks = io.imread(output_mask_fn)
# else:
#     masks = segmenting_vessels_gs_assisted(img, vessel_size_t=2)
#     io.imsave(output_mask_fn, masks.astype(np.uint8))

coords_pixel, coords_cv, coords_pv = find_pv_cv_coords(masks, cv_labels, pv_labels)
orphans = find_orphans(masks, cv_labels, pv_labels)
zone_crit = get_distance_projection(
    coords_pixel, coords_cv, coords_pv, cosine_filter=True
)
zone_crit[masks != 0] = -1
zone_crit[(zone_crit > 1) | (zone_crit < 0)] = -1
zone_crit[orphans] = -1

# coords_pixel, coords_cv, coords_pv = find_pv_cv_coords(masks, cv_labels, pv_labels)
# orphans = find_orphans(masks, cv_labels, pv_labels)
# dist_cv = [ndi.distance_transform_edt(masks != x) for x in cv_labels]
# coord = [
#     ndi.distance_transform_edt(masks != x, return_indices=True, return_distances=False)
#     for x in cv_labels
# ]
# dist_cv, coord = np.array(dist_cv), np.array(coord)
# sorted_dist_cv = np.sort(dist_cv, axis=0).reshape(dist_cv.shape[0], -1)
# dist_cv = dist_cv.reshape(dist_cv.shape[0], -1)

# zone_crit = get_distance_projection(coords_pixel, coords_cv, coords_pv)
# zone_crit[masks != 0] = 0
# out_of_lobule_w_orphan = (zone_crit > 1) | (zone_crit < 0)
# true_out_of_lobule = (
#     np.logical_xor(out_of_lobule_w_orphan, orphans) & out_of_lobule_w_orphan
# )
# io.imshow(true_out_of_lobule, cmap="gray")

# # Find the next nearest cv
# dim1 = np.where(dist_cv == sorted_dist_cv[1, :], dist_cv, np.zeros(dist_cv.shape))
# dim1 = np.apply_along_axis(np.argmax, 0, dim1)
# dim2 = np.arange(dim1.shape[0])
# new_cv_corrds = coord.reshape(32, 2, -1)[dim1, :, dim2]

# # assign previous out of lobule pixels as orphan if the new nearest cv is too far
# new_dist = dist_cv[dim1, dim2].reshape(masks.shape[0], -1)
# orphans = orphans | ((new_dist > 400) & true_out_of_lobule)

# # Updates cv coordinates
# new_cv_corrds = new_cv_corrds.transpose().reshape(coords_cv.shape)
# coords_cv[:, true_out_of_lobule] = new_cv_corrds[:, true_out_of_lobule]

# new_zone_crit = zone_crit.copy()
# new_zone_crit[(new_zone_crit > 1) | (new_zone_crit < 0)] = -30
# new_zone_crit[orphans] = -10
# new_zone_crit[np.isin(masks, cv_labels)] = 20
# new_zone_crit[np.isin(masks, pv_labels)] = 30
# io.imshow(new_zone_crit)

zones = create_zones(masks, zone_crit, cv_labels, pv_labels, num_zones=5)
zone_int = plot_zone_int_probs(
    img[:, :, 0], img[:, :, 2], zones, dapi_cutoff="otsu", plot_type="ppz"
)
zones[zones == -1] = 7
zones[zones == 255] = 6
zones = zones * 20
plt.imshow(img[:, :, 0] > 105)
plt.imshow(zones, alpha=0.5)
