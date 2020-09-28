#%%
# import argparse
from goz.find_zones import *
from goz.plotting import *
from goz.segmentation import *
import os
from skimage import io, filters, measure
import pandas as pd
import warnings
import matplotlib
import sys
#%%
def merge_mask(labeled_mask, max_dist=20):
    # iteratively merging masks until it cannot be merged.
    merged_mask = labeled_mask.copy()
    new_merged_mask = merge_neighboring_vessels(labeled_mask, max_dist)[0]
    while (new_merged_mask != merged_mask).all():
        merged_mask = new_merged_mask
        print("Continue merging neighboring masks...")
        new_merged_mask, _ = merge_neighboring_vessels(merged_mask, max_dist)[0]
    return new_merged_mask

def pool_masks(l_cv_masks, l_pv_masks):
    # merge two non-overlapping lebeled masks together.
    # ensure there is no overlap
    set(np.unique(l_cv_masks))&set(np.unique(l_pv_masks)) == {0}
    masks = l_cv_masks + l_pv_masks
    output_masks = np.zeros(masks.shape, int)
    cv_labels = []
    pv_labels = []
    for label, region in enumerate(measure.regionprops(masks)):
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


#%%
input_folder = '/home2/s190548/work_zhu/Gozdeep/Gozdeep_input'
output_folder = '/home2/s190548/work_zhu/Gozdeep/results'
tifs = [x for x in os.listdir(input_folder) if '_mask.tif' not in x]
design = pd.read_csv(input_folder + '/design.txt', sep='\t', index_col=0)
for outfolder in design.folder.unique():
    if not os.path.exists(output_folder + '/' + outfolder):
        os.mkdir(output_folder + '/' + outfolder)

#%%
for tif in design.index:
    tif_fn = os.path.join(input_folder, tif)
    if not os.path.exists(tif_fn):
        continue
    mask_fn = os.path.join(input_folder, tif.replace('.tif','_mask.tif'))
    spot_size = design.loc[tif,'spot']
    output_prefix = os.path.join(
        output_folder, design.loc[tif,'folder'], tif.strip('.tif'))
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)
    output_prefix += '/'
    masks = io.imread(mask_fn)
    img = io.imread(tif_fn)
    io.imshow(img)
    plt.savefig(output_prefix+'raw_image_scaled.pdf')
    plt.close()
    cv_masks = measure.label(masks[:,:,1]>0)
    pv_masks = measure.label(masks[:,:,2]>0)

    cv_masks = merge_mask(cv_masks)
    pv_masks = merge_mask(pv_masks)
    pv_masks[pv_masks>0] = pv_masks[pv_masks>0] + cv_masks.max()
    masks, cv_labels, pv_labels, cv_masks, pv_masks = pool_masks(
        cv_masks, pv_masks)

    # find lobules
    cv_masks = cv_masks.astype("uint8")
    _, lobules_sizes, lobule_edges = find_lobules(cv_masks)
    plot3channels(
        lobule_edges,
        cv_masks != 0,
        pv_masks != 0,
        fig_name=output_prefix + "lobules",
    )

    # Calculate distance projections
    #! orphan cut off set at 550
    zone_crit = calculate_zone_crit(cv_masks, pv_masks, tolerance=550)

    # Calculate zones
    zones = create_zones(
        masks,
        zone_crit,
        cv_labels,
        pv_labels,
        zone_break_type="equal_quantile",
        num_zones=24,
    )
    plot_zones_only(zones)
    # Plot zones with image
    plot_zone_with_img(img, zones, fig_prefix=output_prefix + "zones with marker")
    plot_zones_only(zones, fig_prefix=output_prefix + "zones only")

    # Calculate zonal reporter expression levels.
    tomato_cutoff = filters.threshold_otsu(img[:,:,0])
    zone_int = plot_zone_int_probs(
        img[:, :, 0],
        img[:, :, 2],
        zones,
        dapi_cutoff="otsu",
        plot_type="probs",
        tomato_cutoff=tomato_cutoff,
        prefix=output_prefix + "Marker",
    )
    zone_int.to_csv(output_prefix + "zone int.csv")

    # Calculate zonal spot clonal sizes.
    if spot_size:
        spot_sizes_df, skipped_boxes, valid_nuclei_mask = calculate_clonal_size(
            img, zones, tomato_erosion=1
        )
        spot_segmentation_diagnosis(
            img,
            spot_sizes_df,
            skipped_boxes,
            valid_nuclei_mask,
            fig_prefix=output_prefix,
        )
        plot_spot_clonal_sizes(
            spot_sizes_df,
            absolute_number=False,
            figname=output_prefix + "spot_clonal_sizes.pdf",
        )
        spot_sizes_df.to_csv(output_prefix + "spot clonal sizes.csv")
# %%
