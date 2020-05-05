import pandas as pd
from skimage import io, morphology, measure
import numpy as np
import os
import matplotlib.pyplot as plt
from goz.segmentation import *
from goz.plotting import *
from goz.find_zones import *
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

#! Note to myself: got the io part figured out, but need to find/make a good way to
#! threshold the channels, the channels need to be thresholded individually.


def find_valid_crops(dapi, cols=3500, rows=1500, padding=250):
    #! row are rows, and columsn are cols
    n_col_steps = int(np.ceil(dapi.shape[1] / cols))
    n_row_steps = int(np.ceil(dapi.shape[0] / rows))
    valid_crops = []
    for col in range(n_col_steps):
        for row in range(n_row_steps):
            img_crop = np.zeros((rows + 2 * padding, cols + 2 * padding))
            box_l = np.max((cols * col - padding, 0))
            box_r = np.min((cols * (col + 1) + padding, dapi.shape[1]))
            box_t = np.max((rows * row - padding, 0))
            box_b = np.min((rows * (row + 1) + padding, dapi.shape[0]))
            img_crop[: box_b - box_t, : box_r - box_l] = dapi[box_t:box_b, box_l:box_r]
            # print(col, row, np.median(img_crop))
            if np.median(img_crop > 0):
                valid_crops.append([box_t, box_b, box_l, box_r])
    return valid_crops


def plot_image_crops(dapi, crop_df, padding, prefix):
    crop_df = crop_df.copy()
    crop_df[crop_df == 0] = -250
    crop_df.iloc[:, [0, 2]] = crop_df.iloc[:, [0, 2]] + padding
    crop_df.iloc[:, [1, 3]] = crop_df.iloc[:, [1, 3]] - padding
    dapi_crop_show = dapi.copy()
    masks = np.zeros(dapi.shape, "bool")
    for _, box in crop_df.iterrows():
        top, bottom, left, right = box
        left = np.max([left, 50])
        top = np.max([top, 50])
        masks[top:bottom, left - 50 : left + 50] = True
        masks[top:bottom, right - 50 : right + 50] = True
        masks[top - 50 : top + 50, left:right] = True
        masks[bottom - 50 : bottom + 50, left:right] = True
    _ = plt.figure(figsize=(16,9))
    io.imshow(masks, cmap="gray")
    io.imshow(dapi, alpha=0.5, cmap="gray")
    plt.savefig(prefix+'image_crops.pdf')
    plt.close()


def find_spans(left, right):
    left = np.array(sorted(left))
    right = np.array(right)[np.argsort(left)]
    spans = []
    _left = left[0]
    _right = right[0]
    for i in range(1, len(left)):
        if left[i] <= _right:
            _right = right[i]
        else:
            spans.append([_left, _right])
            _left = left[i]
            _right = right[i]
    if [_left, _right] not in spans:
        spans.append([_left, _right])
    return spans


def merge_overlapping_boxes(valid_crops):
    crop_df = pd.DataFrame(valid_crops, columns=["t", "b", "l", "r"])
    crop_df = crop_df.sort_values(["t", "b", "l", "r"])
    for horizontal in [True, False]:
        if horizontal:
            loc_types = ["t", "b"]
            loc_left = "l"
            loc_right = "r"
        else:
            loc_types = ["l", "r"]
            loc_left = "t"
            loc_right = "b"

        while True:
            new_crops = []
            # find horizontal grids
            for _group in crop_df.groupby(loc_types):
                crit_left, crit_right = _group[0]
                left, right = [*_group[1].loc[:, loc_left:loc_right].T.values]
                _span = find_spans(left, right)
                # print(horizontal, _group[0])
                # print(_span)
                # print(_group[1].loc[:, loc_left:loc_right])
                # print("********")
                for indiv_span in _span:
                    if horizontal:
                        new_crops.append(
                            [crit_left, crit_right, indiv_span[0], indiv_span[1]]
                        )
                    else:
                        new_crops.append(
                            [indiv_span[0], indiv_span[1], crit_left, crit_right]
                        )
                # print(new_crops)
                # print("-------")
            new_crops = pd.DataFrame(new_crops, columns=crop_df.columns)
            new_crops = new_crops.sort_values(new_crops.columns.tolist())
            new_crops = new_crops.drop_duplicates()
            if new_crops.shape == crop_df.shape:
                if (new_crops.values == crop_df.values).all().all():
                    break
            else:
                crop_df = new_crops
    return crop_df


def pool_masks_from_crops(img, mask_files, img_border_mask_eroded, padding=250):
    overall_masks = np.zeros(img.shape[:2], 'uint32')
    vessels = np.zeros(img.shape[:2], "bool")
    mask_offset = 0
    for fn in mask_files:
        _img = io.imread(fn)
        _vessels = _img[:, :, 1]
        _masks = _img[:, :, 0]
        _masks[_masks>0] = _masks[_masks>0] + mask_offset
        coords = [int(x) for x in fn.split("/")[-1].split("_")[1].split(" ")]
        top, bottom, left, right = coords
        _masks = _masks[padding:-padding, padding:-padding]
        _vessels = _vessels[padding:-padding, padding:-padding] != 0
        overall_masks[
            top + padding : bottom - padding, left + padding : right - padding
        ] = _masks
        vessels[
            top + padding : bottom - padding, left + padding : right - padding
        ] = _vessels
        mask_offset = mask_offset + len(np.unique(_masks)) - 1
    overall_masks = overall_masks * img_border_mask_eroded
    vessels = vessels * img_border_mask_eroded
    # Merge split masks that land near the edge of image crops
    overall_masks_labeled = measure.label(overall_masks!=0,connectivity=2)
    for region in measure.regionprops(overall_masks_labeled):
        x0,y0,x1,y1 = region.bbox
        old_labels = overall_masks[x0:x1,y0:y1][region.image]
        if len(np.unique(old_labels)) > 1:
            old_labels = [str(x) for x in np.unique(old_labels)]
            print(
                'Split mask found and merged: {}'.format(', '.join(old_labels))
                )
            overall_masks[x0:x1,y0:y1][region.image] = old_labels[0]
    return overall_masks, vessels


def mask_pruning(labeled_masks, vessel_size_l, clustering_based_pruning = False):
    """Use topology features of masks and hierarchical clustering to get rid of bad
    masks.
    """
    labeled_masks = labeled_masks.copy()
    props = pd.DataFrame(
        measure.regionprops_table(
            labeled_masks, properties=("label", "area", "extent", "perimeter")
        )
    ).set_index("label")
    props["circularity"] = np.pi * 4 * props.area / np.power(props.perimeter, 2)
    valid_props = props[
        (props.area >= vessel_size_l) & 
        (props.area < 1000*vessel_size_l)
        ].copy()
    print(
        "{} out of {} masks are kept due to size.".format(
            valid_props.shape[0], props.shape[0]
        )
    )
    if clustering_based_pruning:
        valid_props.iloc[:, :2] = np.log2(valid_props.iloc[:, :2])
        valid_props_data = valid_props.copy()
        cluster_estimator = AgglomerativeClustering(
            2, affinity="correlation", linkage="average"
        )
        labels = cluster_estimator.fit_predict(valid_props_data)
        masks_mean_props = valid_props_data.groupby(labels).mean()
        good_label = masks_mean_props.circularity.idxmin()
        good_masks = valid_props_data.index[labels == good_label]
        print(
            "{} out of {} masks are kept".format(len(good_masks), valid_props.shape[0])
        )
        labeled_masks[~np.isin(labeled_masks, good_masks)] = 0
    else:
        good_label = valid_props.index.unique()
        labeled_masks[~np.isin(labeled_masks, good_label)] = 0
    return labeled_masks
