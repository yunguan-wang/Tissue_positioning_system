import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale, minmax_scale
from sklearn.neighbors import NearestNeighbors
from skimage import io, measure, morphology, filters, color, transform, util, feature
import scipy.ndimage as ndi
from sklearn.decomposition import FastICA, PCA

def segmenting_vessels(
    img, dark_t=20, dapi_channel=2, vessel_size_t=2, dapi_dilation_r=10
):
    # use gray scale image as vein signal
    # img_gray = (255*color.rgb2gray(img)).astype('uint8')
    # NOTE allows for single channel input
    if len(img.shape) == 3:
        img_gray = img[:, :, dapi_channel]
    else:
        img_gray = img
    # NOTE This dapi dilation step expands the white area around DAPI and 
    # reduced the noise masks. Useful in large image processing.
    if dapi_dilation_r > 0:
        img_gray = morphology.dilation(img_gray, selem=morphology.disk(dapi_dilation_r))
    veins = img_gray < dark_t
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
    # NOTE recover eroded mask from previous dilation.
    if dapi_dilation_r > 0:
        filled_image = morphology.dilation(
            filled_image, selem=morphology.disk(dapi_dilation_r)
        )
    return filled_image


def extract_features(labeled_mask, raw_gs_ica, q1=0.25, q2=0.75, step=0.05):
    # a simple erosion to get the outside ring of the mask, where GS is mostly 
    # expressed.
    selem = morphology.disk(20)
    eroded = morphology.erosion(labeled_mask, selem)
    labeled_mask = (labeled_mask - eroded).copy()
    # background mask is thrown away here.
    mask_features = pd.DataFrame(index=sorted(np.unique(labeled_mask))[1:])
    for region in measure.regionprops(labeled_mask, raw_gs_ica):
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
        pixel_ints = raw_gs_ica[region.coords[:, 0], region.coords[:, 1]]
        for j, p in enumerate(np.arange(q1, q2, step)):
            mask_features.loc[label, "I" + str(j)] = np.quantile(pixel_ints, p)
        mask_features.loc[label,'median_gs'] = np.median(pixel_ints)
    return mask_features


def pv_classifier(cv_features, labeled_mask, max_cv_pv_ratio = 2):
    model_data = cv_features.copy()
    km = KMeans(2)
    pca = PCA(2, whiten=True,random_state=0)
    adjust_pv = None
    cv_feature_pca = pca.fit_transform(model_data)
    labels = km.fit_predict(cv_feature_pca)
    class_median_int = model_data.groupby(labels).median_gs.median().sort_values()
    high_int_label = class_median_int.index[1]
    cv_labels = model_data.index[labels == high_int_label]
    pv_labels = model_data.index[labels != high_int_label]
    num_cv_old, num_pv_old = len(cv_labels), len(pv_labels)
    # Theoraticall the ration between number of CV and PV should be around 1
    # Guided with this, a backup mechnism is designed to adjust the number of 
    # CV and PV buy assigning additional vessels to the minor vessel class using
    # KNN algorithm. 
    if len(cv_labels) / len(pv_labels) > max_cv_pv_ratio:
        adjust_pv = True
        seed_labels = pv_labels
    elif len(cv_labels) / len(pv_labels) < 1/max_cv_pv_ratio:
        adjust_pv = False
        seed_labels = cv_labels
    else:
        print('Number of CV and PV: {}, {}'.format(num_cv_old, num_pv_old))
        return cv_labels, pv_labels
    # Starting adjusting using KNN.
    if adjust_pv is not None:
        desired_minor_class_size = int(
            len(model_data) * 1/(max_cv_pv_ratio+1)
            ) + 1
        num_new = 0
        n_neighbors = 1
        while len(seed_labels) <= desired_minor_class_size:
            if num_new == 0:
                n_neighbors += 1
            else:
                n_neighbors = 2
            nbrs = NearestNeighbors(
                n_neighbors=n_neighbors,
                algorithm='ball_tree'
                ).fit(cv_feature_pca)
            _, indices = nbrs.kneighbors(cv_feature_pca)
            iloc_idx = model_data.index.isin(seed_labels)
            neighbor_labels = np.unique(indices[iloc_idx].flatten())
            neighbor_labels = cv_features.index[neighbor_labels]
            num_new = len([x for x in neighbor_labels if x not in seed_labels])
            seed_labels = list(set(neighbor_labels)|set(seed_labels))
        if adjust_pv:
            pv_labels = seed_labels
            cv_labels = [x for x in model_data.index if x not in pv_labels]
        else:
            cv_labels = seed_labels
            pv_labels = [x for x in model_data.index if x not in cv_labels]
        num_cv_new, num_pv_new = len(cv_labels), len(pv_labels)
        print('Number of CV adjusted from {} to {}.'.format(
            num_cv_old, num_cv_new))
        print('Number of PV adjusted from {} to {}.'.format(
            num_pv_old, num_pv_new))
    return cv_labels, pv_labels



def extract_gs_channel(img, gs_channel=1):
    """Fix color cross talk by using ICA
    """
    ica = FastICA(3, random_state=0)
    ica_transformed = ica.fit_transform(img.reshape(-1, 3))
    # calculate the correlation between the transformed data with target channel
    # This is the correct way of doing this because neither the mixing nor 
    # unmixing matrix reflects the best GS channel from the transformed data.
    corr = np.corrcoef(
        scale(img.reshape(-1, 3)), scale(ica_transformed), rowvar=False
        )[:3, 3:]
    crit = np.argmax(abs(corr), axis=0) == gs_channel
    # Rare case 1 where the gs channel is not dominant in any components. In 
    # this case, it is better to use the raw channel with thresholding.
    if crit.sum() == 0:
        gs_component = None
    # Rare case 2 where the gs channel is not dominant in more than 1 components
    elif crit.sum() > 1:
        gs_component = np.argmax(abs(corr[:, crit]), axis=1)[gs_channel]
    else:
        gs_component = np.argmax(crit)
    
    # Debugging step to identify the problem
    '''
    gs_component_mixing = np.argmax(
        abs(ica.mixing_).argmax(axis=1) == gs_channel
        )
    gs_component_unmixing = np.argmax(
        abs(ica.components_).argmax(axis=1) == gs_channel
        )
    print(ica.mixing_, gs_component, gs_component_mixing,gs_component_unmixing)
    '''
    if gs_component is not None:
        gs_ica = ica_transformed[:, gs_component]
        gs_ica = minmax_scale(gs_ica) * 255
        gs_ica = gs_ica.reshape(img.shape[:2]).astype(np.uint8)
        if gs_ica.mean() > 128:
            gs_ica = 255 - gs_ica
        raw_gs_ica = gs_ica
        gs_ica = gs_ica > filters.threshold_otsu(gs_ica)
    else:
        gs_ica = img[:,:,gs_channel]
        gs_t = filters.threshold_otsu(gs_ica[gs_ica>10])
        raw_gs_ica = gs_ica
        gs_ica = gs_ica > gs_t
    # Returns ICA processed GS channel for better classification.
    return gs_ica, ica, raw_gs_ica


def merge_neighboring_vessels(labeled_mask, max_dist=10):
    """
    Calculate neares pixel to pixel distance between two nearby masks, if
    they are within the threshold, the masks will be merged.
    """
    # remove the mask for background, which is 0
    all_masks = sorted(np.unique(labeled_mask))[1:]
    dist_to_mask = [
        ndi.distance_transform_edt(labeled_mask != x) for x in all_masks]
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
    gs_added_mask_size_t=1,
    max_dist=10,
    dark_t=20,
    dapi_channel=2,
    vessel_size_t=2,
    gs_channel=1,
    gs_ica=None,
    dapi_dilation_r=0,
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
        dark_t=dark_t,
        dapi_channel=dapi_channel,
        vessel_size_t=vessel_size_t,
        dapi_dilation_r=dapi_dilation_r,
    )
    gs_dapi_mask = morphology.opening(gs_ica,morphology.disk(5)) | (vessels != 0)
    # dilate the new mask a little bit to reduce the gap size.
    gs_dapi_mask = filters.gaussian(gs_dapi_mask, 1) > 0
    # Calculate each pixels distance to pre-existing vessels
    labeled_vessels = measure.label(vessels, connectivity=1)  # labeling
    # get rid of background mask
    all_vessels = sorted(np.unique(labeled_vessels))[1:]
    # Calculate the min distance to each vessels
    dist_to_mask = [
        ndi.distance_transform_edt(
            labeled_vessels != x
            ) for x in np.unique(all_vessels)
        ]
    dist_to_mask = np.array(dist_to_mask)
    min_dist_to_mask = np.min(dist_to_mask, axis=0)  # exact distance
    min_dist_label = (
        np.argmin(dist_to_mask, axis=0) + 1
    )  # To which vessel is the min distance observed

    labeled = measure.label(gs_dapi_mask, connectivity=2)
    size_cutoff = gs_added_mask_size_t * img.shape[0] * img.shape[1] / 10000
    # Make newly formed GS masks near existing vessel masks inherit the vessel 
    # labels.
    merged_mask = labeled_vessels
    new_label = np.max(labeled_vessels) + 1
    for region in measure.regionprops(labeled):
        # Get minimal distance, and the vessel label associated with such 
        # minimal distnce
        _region_mask = labeled == region.label
        _min_dist_to_mask = min_dist_to_mask[_region_mask].min()
        _min_dist_coord_idx = np.argmin(min_dist_to_mask[_region_mask])
        _min_dist_coord = region.coords[_min_dist_coord_idx]
        _min_dist_vessel_label = min_dist_label[
            _min_dist_coord[0], _min_dist_coord[1]
            ]
        # If the GS mask is sufficiently close to a vessel mask, the GS masks 
        # share the vessel label
        if _min_dist_to_mask < max_dist:
            merged_mask[_region_mask] = _min_dist_vessel_label
        else:
            if region.filled_area > size_cutoff:
                x0, y0, x1, y1 = region.bbox
                merged_mask[_region_mask] = new_label
                new_label += 1
    # now, merge neighboring masks
    print("Merging neighboring masks...")
    new_merged_mask, _ = merge_neighboring_vessels(
        merged_mask, max_dist=max_dist)
    while not (new_merged_mask == merged_mask).all():
        merged_mask = new_merged_mask
        print("Continue merging neighboring masks...")
        new_merged_mask, _ = merge_neighboring_vessels(merged_mask, max_dist=max_dist)

    # Getting rid of very small masks formed by GS and fillholes
    selem = morphology.disk(10)
    new_merged_mask = morphology.closing(new_merged_mask, selem)

    masks = np.zeros(new_merged_mask.shape, dtype=np.int64)
    for region in measure.regionprops(new_merged_mask):
        if region.area > size_cutoff:
            label = region.label
            x0, y0, x1, y1 = region.bbox
            filled = region.filled_image
            masks[x0:x1, y0:y1][filled] = label

    return masks, raw_gs_ica, vessels


def shrink_cv_masks(
    labeled_cv_masks, labeled_pv_masks, vessels, keep_non_vesseled_gs=True
):
    """Shrink masks so that they does not include the GS positive layer, 
    which is used in earliers steps for CV PV classification.

    Parameters
    ========
    labeled_cv_masks, labeled_pv_masks : np.array
        labeled image masks for cv and pv, which combined is the original masks to be shrank.
    vessels : np.array
        image masks for the vessel-like holes only, which is the real vessel mask.
    keep_non_vesseled_gs : bool
        whether treat gs-positive masks with no vessel-like holes as a CV mask, this is useful
        only in handling large image processing, in small images, this should always be True.

    Returns
    ========
    new_masks : np.array
        Shrank image masks of the same shape and original labels.
    """
    new_masks = np.zeros(labeled_cv_masks.shape)
    if vessels.dtype != "bool":
        vessels = vessels != 0
    # shrinking CV masks
    for _masks, _mask_type in zip(
        [labeled_cv_masks, labeled_pv_masks], ["cv", "pv"]
        ):
        for region in measure.regionprops(_masks):
            if region.label == 0:
                continue
            label = region.label
            min_row, min_col, max_row, max_col = region.bbox
            area = region.area
            area_vessel = vessels[min_row:max_row, min_col:max_col].sum()
            area_image = region.image
            vessel_image = vessels[min_row:max_row, min_col:max_col] & region.image
            if new_masks[min_row:max_row, min_col:max_col][area_image].sum() != 0:
                print(new_masks[min_row:max_row, min_col:max_col][area_image].unique())
                raise ValueError
            if (
                (_mask_type == "cv")
                & (area_vessel / area < 0.05)
                & (keep_non_vesseled_gs)
            ):
                # if the mask is a CV mask and there is a drastic reduction when look
                # at vessel mask, it means the mask is still a CV mask but the vessel
                # if not sliced, thus we keep the whole CV mask
                selem = morphology.disk(25)
                area_image = morphology.binary_erosion(area_image,selem)
                if area_image.sum() < 1000:
                    continue
                new_masks[min_row:max_row, min_col:max_col][area_image] = label
            else:
                # otherwise, only the vesseled part will be kept
                new_masks[min_row:max_row, min_col:max_col][vessel_image] = label
    return new_masks


def find_boundry(dapi):
    # x10, y10 = [int(x/10) for x in dapi.shape]
    # img_10th = transform.resize(dapi,(x10,y10))
    # x100, y100 = [int(x/10) for x in img_10th.shape]
    # img_100h = transform.resize(img_10th,(x100,y100))
    # img_100h_padded = util.pad(img_100h,10)
    # edge_t = filters.threshold_otsu(img_100h_padded)
    # edges = feature.canny(img_100h_padded>edge_t,sigma=3)
    # edges = morphology.closing(edges,morphology.disk(3))
    # dapi_boundry = ndi.binary_fill_holes(edges)
    # dapi_boundry = dapi_boundry[10:-10,10:-10]
    # dapi_boundry = transform.resize(dapi_boundry, (x10,y10))
    # dapi_boundry = transform.resize(dapi_boundry, (dapi.shape))
    # dapi_boundry = dapi_boundry
    boundry_masks = filters.gaussian(dapi>50, 4)*1.0
    boundry_masks = filters.gaussian(boundry_masks>0, 4)*1.0
    boundry_masks = filters.gaussian(boundry_masks>0, 4)*1.0
    labeled_boundry_masks = measure.label(boundry_masks)
    rps_table = pd.DataFrame(measure.regionprops_table(
        labeled_boundry_masks, properties=['label','area']))
    mask_label = rps_table.loc[rps_table.area.idxmax(),'label']
    boundry_masks = labeled_boundry_masks == mask_label
    boundry_masks = ndi.binary_fill_holes(boundry_masks)
    img_border_mask_eroded = boundry_masks.copy()
    for i in range(50):
        img_border_mask_eroded = morphology.binary_erosion(
            img_border_mask_eroded, morphology.disk(2))
    return img_border_mask_eroded
