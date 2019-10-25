import pandas as pd
import skimage as ski
from skimage import io
import os
from segmentation import *
from plotting import *
from find_zones import *

os.chdir("T:/images/New Tif/Normal  30%")
tif_files = sorted([x for x in os.listdir() if ".tif" in x])[::-1][26:]
for img_fn in tif_files:
    output_prefix = img_fn.replace(".tif", "")
    output_mask_fn = output_prefix + " masks.tif"
    img = io.imread(img_fn)
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)
    os.chdir(output_prefix)
    print(output_prefix, output_mask_fn)

    if os.path.exists(output_mask_fn):
        print("Use existing masks")
        masks = io.imread(output_mask_fn)
    else:
        print("Segmentating using GS and DAPI")
        masks = segmenting_vessels_gs_assisted(img, vessel_size_t=2)
        io.imsave(output_mask_fn, masks.astype(np.uint8))
    os.chdir("../")
    # get CV PV classification
    cv_features = extract_features(masks, img[:, :, 1], q1=0.75, q2=0.99, step=0.1)
    cv_labels, pv_labels = pv_classifier(cv_features.loc[:, "I0":], masks)
    # modify CV masks to shrink their borders
    cv_masks = masks * np.isin(masks, cv_labels).copy()
    pv_masks = masks * np.isin(masks, pv_labels).copy()
    masks = shrink_cv_masks(cv_masks, pv_masks, img[:, :, 2])
    plot_pv_cv(masks, cv_labels, img, output_prefix + " ")
    # Calculate distance projections
    coords_pixel, coords_cv, coords_pv = find_pv_cv_coords(masks, cv_labels, pv_labels)
    orphans = find_orphans(masks, cv_labels, pv_labels)
    zone_crit = get_distance_projection(
        coords_pixel, coords_cv, coords_pv, cosine_filter=True
    )
    zone_crit[masks != 0] = -1
    zone_crit[(zone_crit > 1) | (zone_crit < 0)] = -1
    zone_crit[orphans] = -1
    # Calculate zones
    zones = create_zones(masks, zone_crit, cv_labels, pv_labels, num_zones=24)
    _ = plot_zone_int_probs(
        img[:, :, 0],
        img[:, :, 2],
        zones,
        dapi_cutoff="otsu",
        plot_type="probs",
        prefix="output_prefix",
    )
    plot_zone_with_img(img, zones, fig_prefix=output_prefix)
    os.chdir("../")