import pandas as pd
import skimage as ski
from skimage import io
import os
from segmentation import *
from plotting import *
from find_zones import *

os.chdir("T:/images/TIF/")
for img_fn in os.listdir():
    os.mkdir(img_fn.replace(".tif", ""))
    img = io.imread(img_fn)
    os.chdir(img_fn.replace(".tif", ""))
    img = np.array(img, dtype=np.uint8)
    output_mask_fn = img_fn.replace("-1.tif", "_masks.tif")
    output_prefix = img_fn.replace("-1.tif", "")
    if os.path.exists(output_mask_fn):
        masks = io.imread(output_mask_fn)
    else:
        masks = segmenting_vessels_gs_assisted(img, vessel_size_t=2)
        io.imsave(output_mask_fn, masks.astype(np.uint8))

    gs_features = extract_features(masks, img[:, :, 1], q1=0.75, q2=0.99, step=0.1)
    gs_labels, non_gs_labels = pv_classifier(gs_features.loc[:, "I0":], masks)
    plot_pv_cv(masks, gs_labels, img, output_prefix + " ")
    zone_crit = get_distance_projection(masks, gs_labels, non_gs_labels)
    orphans = find_orphans(masks, gs_labels, non_gs_labels)
    zone_crit[zone_crit > 1] = -1
    zone_crit[zone_crit < 0] = -1
    zone_crit[orphans] = -1
    zones = create_zones(masks, zone_crit, num_zones=10)
    io.imsave(output_prefix + "_zone.png", zones)

    _ = plot_zone_int(
        img[:, :, 0],
        img[:, :, 2],
        zones,
        plot_type="violin",
        savefig=True,
        prefix=output_prefix,
        marker_name=output_prefix,
    )
    os.chdir("../")

plt.imshow(img)
plt.imshow(zones, cmap="gray", alpha=0.7)

