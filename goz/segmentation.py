import pandas as pd
import skimage as ski
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale, minmax_scale
import skimage.measure as measure
import skimage.feature as feature
import skimage.filters as filters
import skimage.segmentation as segmentation
import skimage.morphology as morphology
import scipy.ndimage as ndi
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA


def segmenting_vessels(img, dark_t=20, dapi_channel=2, vessel_size_t=2, dilation_t=15):
    veins = img[:, :, dapi_channel] < dark_t
    # Opening operation
    selem = morphology.disk(9)
    opened = morphology.erosion(veins, selem)
    closed = morphology.binary_dilation(opened, selem)
    # getting rid of small dark areas based on area
    size_cutoff = vessel_size_t * img.shape[0] * img.shape[1] / 10000
    labeled = measure.label(closed, connectivity=1)
    filled_image = np.zeros(labeled.shape, dtype=bool)
    for region in measure.regionprops(labeled):
        # Todo: is there better way of setting this threshold?
        if region.area > size_cutoff:
            x0, y0, x1, y1 = region.bbox
            filled_image[x0:x1, y0:y1] |= region.filled_image
    if dilation_t != 0:
        new_mask = morphology.binary_dilation(
            filled_image.astype(int), morphology.disk(dilation_t)
        )
        new_mask = new_mask.astype(int) - filled_image.astype(int)
        labeled_mask = measure.label(new_mask)
        return labeled_mask
    else:
        return filled_image


def extract_features(labeled_mask, img, q1=0.75, q2=1, step=0.05):
    # a simple erosion to get the outside ring of the mask, where GS is mostly expressed.
    eroded = ski.morphology.erosion(labeled_mask, ski.morphology.disk(5))
    labeled_mask = (labeled_mask - eroded).copy()
    # background mask is thrown away here.
    mask_features = pd.DataFrame(index=sorted(np.unique(labeled_mask))[1:])
    for region in measure.regionprops(labeled_mask, img):
        label = region.label
        mask_features.loc[label, "eccentricity"] = region.eccentricity
        mask_features.loc[label, "perimeter"] = region.perimeter
        mask_features.loc[label, "solidity"] = region.solidity
        # mask_features.loc[label,'euler_number'] = region.euler_number
        mask_features.loc[label, "equivalent_diameter"] = region.equivalent_diameter
        mask_features.loc[label, "filled_area"] = region.filled_area
        mask_features.loc[label, "area"] = region.area
        mask_features.loc[label, "centroid"] = "_".format(region.centroid)
        mask_features.loc[label, "mean_intensity"] = region.mean_intensity
        pixel_ints = img[region.coords[:, 0], region.coords[:, 1]]
        for j, p in enumerate(np.arange(q1, q2, step)):
            mask_features.loc[label, "I" + str(j)] = np.quantile(pixel_ints, p)
    return mask_features


def pv_classifier(cv_features, labeled_mask):
    km = KMeans(2)
    pca = PCA(2)
    gs_pca = pca.fit_transform(scale(cv_features))
    labels = km.fit_predict(gs_pca)
    class_median_int = cv_features.groupby(labels).I0.median().sort_values()
    high_int_label = class_median_int.index[1]
    cv_labels = cv_features.index[labels == high_int_label]
    non_gs_labels = cv_features.index[labels != high_int_label]
    return cv_labels, non_gs_labels


def extract_gs_channel(img, gs_channel=1):
    """Fix color cross talk be using ICA
    """
    # Todo still not working all the time. Need to fix it
    ica = FastICA(3, random_state=0)
    ica_transformed = ica.fit_transform(img.reshape(-1, 3))
    # calculate the correlation between the transformed data with target channel.
    # This is the correct way of doing this because neither the mixing nor unmixing
    # matrix reflects the best GS channel from the transformed data.
    corr = np.corrcoef(img.reshape(-1, 3), ica_transformed, rowvar=False)[:3, 3:]
    crit = np.argmax(abs(corr), axis=0) == gs_channel
    # Rare case 1 where the gs channel is not dominant in any components
    if crit.sum() == 0:
        gs_component = np.argmax(abs(corr), axis=1)[gs_channel]
    # Rare case 2 where the gs channel is not dominant in more than 1 components
    elif crit.sum() > 1:
        gs_component = np.argmax(abs(corr[:, crit]), axis=1)[gs_channel]
    else:
        gs_component = np.argmax(crit)
    # Debugging step to identify the problem
    # gs_component_mixing = np.argmax(abs(ica.mixing_).argmax(axis=1) == gs_channel)
    # gs_component_unmixing = np.argmax(abs(ica.components_).argmax(axis=1) == gs_channel)
    # print(ica.mixing_, gs_component, gs_component_mixing,gs_component_unmixing)
    gs_ica = ica_transformed[:, gs_component]
    gs_ica = minmax_scale(gs_ica) * 255
    gs_ica = gs_ica.reshape(img.shape[:2]).astype(np.uint8)
    if gs_ica.mean() > 128:
        gs_ica = 255 - gs_ica
    # Returns ICA processed GS channel for better classification.
    raw_gs_ica = gs_ica
    gs_ica = gs_ica > ski.filters.threshold_otsu(gs_ica)
    return gs_ica, ica, raw_gs_ica


def merge_neighboring_vessels(labeled_mask, max_dist=10):
    """
    Calculate neares pixel to pixel distance between two nearby masks, if
    they are within the threshold, the masks will be merged.
    """
    # remove the mask for background, which is 0
    all_masks = sorted(np.unique(labeled_mask))[1:]
    dist_to_mask = [ndi.distance_transform_edt(labeled_mask != x) for x in all_masks]
    dist_to_mask = np.array(dist_to_mask)
    crit = dist_to_mask < max_dist
    nearest_neighbors = []
    for i, mask_name in enumerate(all_masks):
        if mask_name == 0:
            continue
        _crit = crit[i, :, :]
        _nearest_neighbors = np.unique(labeled_mask[_crit])
        # getting rid of self and background
        _nearest_neighbors = list(set(_nearest_neighbors) - set([0, mask_name]))
        if _nearest_neighbors != []:
            for j in _nearest_neighbors:
                # find all sublists that contains the current 'mask_name' or neighbor 'j'
                existing_item = [
                    x for x in nearest_neighbors if (j in x) or (mask_name in x)
                ]
                if existing_item == []:
                    # if there is none, make a new neiboring mask sublist.
                    nearest_neighbors.append([mask_name, j])
                elif len(existing_item) > 1:
                    # this is to handle a special case where there are two sublits in the
                    # 'existing_item', where one of them contain current 'mask_name' and another
                    # contains the neighbor j. These two sublists will need to be merged.
                    new_item = []
                    for item in existing_item:
                        nearest_neighbors.remove(item)
                        new_item += item
                    new_item = list(set(new_item))
                    nearest_neighbors.append(new_item)
                else:
                    # when there is only one sublist in the 'existing_item' list.
                    existing_item = existing_item[0]
                    if j in existing_item:
                        continue
                    idx = nearest_neighbors.index(existing_item)
                    nearest_neighbors[idx] = nearest_neighbors[idx] + [j]

    new_labeled_masks = np.copy(labeled_mask)
    for similar_masks in nearest_neighbors:
        new_labeled_masks[np.isin(new_labeled_masks, similar_masks)] = similar_masks[0]
    for new_id, old_id in enumerate(np.unique(new_labeled_masks)):
        new_labeled_masks[new_labeled_masks == old_id] = new_id
    return new_labeled_masks, nearest_neighbors


def segmenting_vessels_gs_assisted(
    img,
    size_cutoff_factor=1,
    max_dist=10,
    dark_t=20,
    dapi_channel=2,
    vessel_size_t=2,
    gs_channel=1,
    gs_ica=None,
):
    """
    Segmentation of vessels with both dapi channel information and gs
    channel information.
    The use of gs channel is important because sometimes there is not
    visible dapi holes in the image but strong gs staining, indicating a
    presence of central vein which is not sliced in the slide. By adding
    visible dapi holes in the image but strong gs staining, indicating a
    presence of central vein which is not sliced in the slide. By adding
    the gs channel information, such vessel can be recovered.
    """
    if gs_ica is None:
        gs_ica, _, raw_gs_ica = extract_gs_channel(img, gs_channel=gs_channel)
    else:
        raw_gs_ica = gs_ica.copy()
    vessels = segmenting_vessels(
        img,
        dilation_t=0,
        dark_t=dark_t,
        dapi_channel=dapi_channel,
        vessel_size_t=vessel_size_t,
    )
    gs_dapi_mask = gs_ica | (vessels != 0)
    # dilate the new mask a little bit to reduce the gap size.
    # Todo, this step could be improved? Not really necessary?
    selem = morphology.disk(1)
    gs_dapi_mask = morphology.binary_dilation(gs_dapi_mask, selem)
    # Calculate each pixels distance to pre-existing vessels
    labeled_vessels = measure.label(vessels, connectivity=1)  # labeling
    all_vessels = sorted(np.unique(labeled_vessels))[1:]  # get rid of background mask
    # Calculate the min distance to each vessels
    dist_to_mask = [
        ndi.distance_transform_edt(labeled_vessels != x) for x in np.unique(all_vessels)
    ]
    dist_to_mask = np.array(dist_to_mask)
    min_dist_to_mask = np.min(dist_to_mask, axis=0)  # exact distance
    min_dist_label = (
        np.argmin(dist_to_mask, axis=0) + 1
    )  # To which vessel is the min distance observed

    # Iterate through each combined mask from GS and DAPI, evaluate the minimal distance of each mask
    # to pre-existing vessel, if the distance is very small, merge the mask with existing vessel mask.
    # Todo
    #! Important, I did not differentiate PV from CV here, thus the code might merge a GS intensive region
    #! with a nearby PV.
    labeled = measure.label(gs_dapi_mask, connectivity=2)
    size_cutoff = size_cutoff_factor * img.shape[0] * img.shape[1] / 10000
    # inherit the vessel labels
    merged_mask = labeled_vessels
    new_label = np.max(labeled_vessels) + 1
    for region in measure.regionprops(labeled):
        # Get minimal distance, and the vessel label associated with such minimal distnce
        _region_mask = labeled == region.label
        _min_dist_to_mask = min_dist_to_mask[_region_mask].min()
        _min_dist_coord_idx = np.argmin(min_dist_to_mask[_region_mask])
        _min_dist_coord = region.coords[_min_dist_coord_idx]
        _min_dist_vessel_label = min_dist_label[_min_dist_coord[0], _min_dist_coord[1]]
        if _min_dist_to_mask < max_dist:
            merged_mask[_region_mask] = _min_dist_vessel_label
        else:
            if region.filled_area > size_cutoff:
                x0, y0, x1, y1 = region.bbox
                merged_mask[_region_mask] = new_label
                new_label += 1
    # now, merge neighboring masks
    print("Merging neighboring masks...")
    new_merged_mask, _ = merge_neighboring_vessels(merged_mask, max_dist=max_dist)
    while not (new_merged_mask == merged_mask).all():
        merged_mask = new_merged_mask
        print("Continue merging neighboring masks...")
        new_merged_mask, _ = merge_neighboring_vessels(merged_mask, max_dist=max_dist)
    # Returning not only masks, but also GS_ICA channels.
    return new_merged_mask, raw_gs_ica, vessels


def shrink_cv_masks(cv_masks, pv_masks, vessels):
    new_masks = np.zeros(cv_masks.shape)
    vessel_mask = vessels != 0
    # shrinking CV masks
    for _mask in np.unique(cv_masks):
        if _mask == 0:
            continue
        else:
            new_mask_size = ((cv_masks == _mask) & vessel_mask).sum()
            old_mask_size = (cv_masks == _mask).sum()
            if new_mask_size < 0.01 * old_mask_size:
                new_masks[cv_masks == _mask] = _mask
            else:
                # if the new mask is not drastically reduced, meaning it is a vesseled mask
                # only the vesseled part will be kept
                new_masks[(cv_masks == _mask) & vessel_mask] = _mask

    # shrinking PV masks
    for _mask in np.unique(pv_masks):
        if _mask == 0:
            continue
        else:
            new_mask_size = ((pv_masks == _mask) & vessel_mask).sum()
            old_mask_size = (pv_masks == _mask).sum()
            if new_mask_size < 0.01 * old_mask_size:
                new_masks[pv_masks == _mask] = _mask
            else:
                # if the new mask is not drastically reduced, meaning it is a vesseled mask
                # only the vesseled part will be kept
                new_masks[(pv_masks == _mask) & vessel_mask] = _mask
    return new_masks
