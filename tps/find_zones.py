import pandas as pd
from skimage import measure, morphology, filters, feature, segmentation
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale, minmax_scale
import scipy.ndimage as ndi
from tps.segmentation import merge_neighboring_vessels


def find_pv_cv_coords(masks, cv_labels, pv_labels):
    """
    Improved distance calculation algorithm aimed to get a distance measure that are
    linear to the observed distance to cv and pv.
    For each image pixel, Find points on nearest pv or cv masks, Vpv and Vcv, repectively.
    Parameters
    ========
    masks : np.array
        image mask array, must be labeled.
    (non_)cv_labels : list-like
        labels for (portal) central vein.
    cosine_filter : bool
        determines if the case where vector_cv and vector_pv should be ignored.

    Returns
    ========
    coords_pixel, coords_cv, coords_pv : numpy array of coordinates of each pixel,
    nearest central vein and nearest portal vein
    """
    # get pixel coords matrix
    coords_pixel = np.array(
        np.meshgrid(np.arange(masks.shape[0]), np.arange(masks.shape[1]), indexing="ij")
    )
    # nearest Vcv and Vpv.
    _, coords_cv = ndi.morphology.distance_transform_edt(
        ~np.isin(masks, cv_labels), return_indices=True
    )
    _, coords_pv = ndi.morphology.distance_transform_edt(
        ~np.isin(masks, pv_labels), return_indices=True
    )
    return coords_pixel, coords_cv, coords_pv


def get_distance_projection(coords_pixel, coords_cv, coords_pv, cosine_filter=False):
    """
    Improved distance calculation algorithm aimed to get a distance measure that are
    linear to the observed distance to cv and pv.
    Defines three vectors for each interested pixel (Vi) in the following steps.
    1. find points on nearest pv or cv masks, Vpv and Vcv, repectively.
    2. define these three vectors.
        #! vector_pv: vector from Vi to Vpv
        #! vector_cv: vector from Vcv to Vi
        #! vector_cv_to_pv: vector from Vcv to Vpv
    3. defines distance as:
        #! dot(vector_cv, vector_cv_to_pv)/ ||vector_cv_to_pv||^2
    4. #Todo: decide if direction fileter should be used.
        Since the relative positions of the three points have some limitations.

    Parameters
    ========
    coords_pixel : np.array
        image pixel coordinate array
    coords_cv : np.array
        coordinates of the nearest point to a central vein mask for each pixel
    coords_pv : np.array
        coordinates of the nearest point to a portal vein mask for each pixel.
    """
    # calculate the three vectors.
    vector_cv = coords_pixel - coords_cv
    vector_pv = coords_pv - coords_pixel
    vector_cv_to_pv = coords_pv - coords_cv
    # most important function, get dot product of the two vectors.
    # this is not hard to understand, but tricky to implement without numpy eisum function.
    vector_dot = np.einsum("ijk,ijk->jk", vector_cv, vector_cv_to_pv)
    vector_pv_to_cv_norm = np.linalg.norm(vector_cv_to_pv, axis=0)
    projection = vector_dot / vector_pv_to_cv_norm / vector_pv_to_cv_norm
    if cosine_filter:
        vector_cv_norm = np.linalg.norm(vector_cv, axis=0)
        vector_pv_norm = np.linalg.norm(vector_pv, axis=0)
        cosine_vectors = (
            np.einsum("ijk,ijk->jk", vector_cv, vector_pv)
            / vector_cv_norm
            / vector_pv_norm
        )
        projection[cosine_vectors < 0] = -1
    return projection


def fill_hollow_masks(hollow_labeled_masks):
    filled_labeld_mask = hollow_labeled_masks
    for region in measure.regionprops(hollow_labeled_masks):
        x0, y0, x1, y1 = region.bbox
        filled_labeld_mask[x0:x1, y0:y1] |= region.filled_image * region.label
    return filled_labeld_mask


def dist_to_nn_masks(labeled_mask, target_labels, fill_mask=False, dist=None, nn=3):
    """Get average distance per pixel to nearest n masks."""
    if dist is None:
        if fill_mask:
            filled_labeld_mask = fill_hollow_masks(labeled_mask)
        else:
            filled_labeld_mask = labeled_mask
        dist = [
            ndi.distance_transform_edt(filled_labeld_mask != x) for x in target_labels
        ]
        dist = np.array(dist)
        dist = np.sort(dist, axis=0)
    # get mean distance to nearest n masks
    mean_dist = np.mean(dist[:nn, :, :], axis=0)
    return dist, mean_dist


# --------
# this function is currently no longer used.
def calculate_pv_to_cv_dist_ratio(labeled_mask, pv_masks, cv_masks, nn=1):
    _, dist_to_pv = dist_to_nn_masks(labeled_mask, pv_masks, nn=nn)
    _, dist_to_cv = dist_to_nn_masks(labeled_mask, cv_masks, nn=nn)
    filled_labeled_mask = fill_hollow_masks(labeled_mask)
    dist_to_pv[np.isin(filled_labeled_mask, pv_masks)] = 0
    dist_to_cv[np.isin(filled_labeled_mask, cv_masks)] = np.inf
    dist_ratio = dist_to_pv / dist_to_cv
    dist_ratio = np.log2(1 + dist_ratio)
    dist_ratio[np.isin(filled_labeled_mask, cv_masks)] = dist_ratio.max()
    return dist_ratio


# --------

# This function is not used
# def make_zone_masks(dist_ratio, n_zones, method="division"):
#     zone_mask = np.zeros(dist_ratio.shape)
#     if method == "histogram":
#         for zone in range(n_zones):
#             zone_mask[(dist_ratio > bins[zone]) & (dist_ratio < bins[zone + 1])] = int(
#                 zone + 1
#             )
#     elif method == "division":
#         interval = np.percentile(dist_ratio, 99) / n_zones
#         for i in range(n_zones):
#             zone_mask[
#                 (dist_ratio > i * interval) & (dist_ratio < (1 + i) * interval)
#             ] = int((i + 1))
#         zone_mask[dist_ratio >= np.percentile(dist_ratio, 99)] = int(i + 2)
#     return zone_mask


def find_orphans(masks, cv_labels, pv_labels, orphan_crit=400):
    dist_to_pv = ndi.morphology.distance_transform_edt(~np.isin(masks, pv_labels))
    dist_to_cv = ndi.morphology.distance_transform_edt(~np.isin(masks, cv_labels))
    orphans = (dist_to_pv > orphan_crit) | (dist_to_cv > orphan_crit)
    return orphans

def calculate_zone_crit(cv_masks, pv_masks, tolerance=50):
    cv_dist = ndi.distance_transform_edt(cv_masks == 0)
    pv_dist = ndi.distance_transform_edt(pv_masks == 0)
    cv_zones = np.zeros(cv_masks.shape, "uint8")
    pv_zones = np.zeros(cv_masks.shape, "uint8")
    # Iteratively expand a radius cutoff to assign regions around CV or PV into
    # bands of different distance.
    for _dist, _zone, _destination_mask in zip(
        [cv_dist, pv_dist], [cv_zones, pv_zones], [pv_masks, cv_masks]
    ):
        updatable = ~np.zeros(_dist.shape, "bool")
        i = 1
        while True:
            if updatable.sum() == 0:
                print("All image region covered, stop expansion")
                break
            elif ((_destination_mask != 0) * updatable).sum() == 0:
                print("All opposite masks covered, stop expansion")
                break
            elif i > tolerance:
                print("Max distance reached")
                break
            r = i * 10
            _expansion_mask = np.logical_and(updatable, _dist < r)
            _zone[_expansion_mask] = i
            updatable[_expansion_mask] = False
            i += 1
    zone_crit = np.zeros(updatable.shape)
    orphans = np.logical_or(cv_zones == 0, pv_zones == 0)
    zone_crit[~orphans] = np.log2(cv_zones[~orphans] / pv_zones[~orphans])
    zone_crit = (zone_crit - zone_crit.min()) / (zone_crit.max() - zone_crit.min())
    zone_crit[orphans] = 0
    return zone_crit

def create_zones(
    masks, zone_crit, cv_labels, pv_labels, zone_break_type="equal_length", num_zones=5
):
    # CV are labeled as -1
    # PV are labeled as 255
    zones = np.zeros(masks.shape[:2])
    # ignore zone_crit from masks.
    valid_zone_crit = zone_crit.copy()
    valid_zone_crit[masks!=0] = 0
    if zone_break_type == "equal_length":
        for i in range(num_zones):
            t0 = i / num_zones
            t1 = (i + 1) / num_zones
            if i == num_zones - 1:
                zones[zone_crit > t0] = i + 1
            else:
                zones[(zone_crit > t0) & (zone_crit <= t1)] = i + 1
    elif zone_break_type == "equal_quantile":
        quantiles = np.linspace(0, 1, num_zones + 1)
        # adjusting first and last zone so that they are not too big
        diff = quantiles[1]/2
        quantiles[1] = quantiles[1] - diff
        quantiles[-2] = quantiles[-2] + diff
        j=1
        for i in range(num_zones-3):
            quantiles[i+2] = quantiles[i+2] - quantiles[1] + 2*diff/(num_zones-2)*j
            j+=1
        #
        zone_breaks = np.quantile(valid_zone_crit[valid_zone_crit>0], quantiles)
        for i, zone_break in enumerate(zone_breaks[:-1]):
            zones[
                (valid_zone_crit > zone_break) & 
                (valid_zone_crit <= zone_breaks[i + 1])
                ] = i + 1
    zones[valid_zone_crit==0] = 0
    zones[np.isin(masks, cv_labels)] = -1
    zones[np.isin(masks, pv_labels)] = 255
    return zones


def find_lobules(cv_masks, outlier_t=0.1, lobule_name="lobule"):
    """Find lobules based on watershed on known CV masks.

    Returns the lobule masks and lobule sizes.
    """
    cv_dist = ndi.distance_transform_edt(cv_masks == 0)
    # get centroid of each CV, use it as the watershed peaks.
    markers = np.zeros(cv_masks.shape)
    for region in measure.regionprops(cv_masks):
        x_c, y_c = region.centroid
        cv_name = region.label
        markers[int(x_c), int(y_c)] = cv_name
    # watershed
    lobules = morphology.watershed(cv_dist, markers)
    # Find lobule boundaries via evaluating pixel greadients.
    grads = filters.rank.gradient(lobules, morphology.disk(5))
    lobule_edges = grads != 0
    # calculating lobule sizes
    lobule_sizes = pd.DataFrame()
    for region in measure.regionprops(lobules):
        lobule_sizes.loc[region.label, "lobule_size"] = region.area
    cutoff_low = lobule_sizes.lobule_size.quantile(outlier_t)
    cutoff_high = lobule_sizes.lobule_size.quantile(1 - outlier_t)
    lobule_sizes = lobule_sizes[
        (lobule_sizes.lobule_size >= cutoff_low)
        & (lobule_sizes.lobule_size <= cutoff_high)
    ]
    lobule_sizes = np.sqrt(lobule_sizes)
    lobule_sizes["lobule_name"] = lobule_name
    return lobules, lobule_sizes, lobule_edges

def watershed_masks(unlabelled_mask,
    dapi_mask = None,
    min_distance=None,
    num_peaks_per_label=999,
    footprint = None,
    ):
    '''
    segmenting markers masks, or dapi masks.
    '''
    # Still in testing phase.
    distance = ndi.distance_transform_edt(unlabelled_mask)
    local_maxi = feature.peak_local_max(
        distance,
        indices=False,
        labels=morphology.label(unlabelled_mask),
        min_distance=min_distance,
        num_peaks_per_label=num_peaks_per_label,
        footprint=footprint
        )
    if dapi_mask is not None:
        dapi_mask = morphology.opening(dapi_mask,morphology.disk(1))
        dapi_mask = morphology.label(dapi_mask)
        merged_dapi_mask = merge_neighboring_vessels(dapi_mask,10)[0]
        markers = np.zeros(merged_dapi_mask.shape,'uint8')
        i = 1
        for region in measure.regionprops(merged_dapi_mask):
            cent_x, cent_y = region.centroid
            markers[int(cent_x),int(cent_y)] = i
            i+=1
    else:
        markers = ndi.label(local_maxi)[0]
    labels = segmentation.watershed(-distance, markers, mask=unlabelled_mask)
    return labels

def calculate_clonal_size(img, zones, tomato_erosion=5, max_nuclei_dist=10):
    """
    Calculate the clonal size of each marker spots in zone. Will try to get rid of marker spots with
    no nuclei or connected-but-not-from-the-same-clone.
    """
    int_img = img[:, :, 0].copy()
    dapi = img[:, :, 2].copy()
    # dapi and marker threshold set by OTSU.
    # dapi_t = filters.threshold_otsu(dapi)
    # print('DAPI threshold for cells detection: {}'.format(dapi_t))
    int_cutoff = filters.threshold_otsu(
        int_img[(int_img < 255) & (int_img > 0)])
    int_signal_mask = int_img > int_cutoff
    int_signal_mask = morphology.erosion(
        int_signal_mask, morphology.disk(tomato_erosion))
    labeled_int_signal_mask = morphology.label(int_signal_mask, connectivity=1)
    zones[(zones < 0) | (zones == 255)] = 0
    # initialize
    clonal_sizes = []
    spot_sizes = []
    zone_lables = []
    parent_bbox = []
    processed_bboxes = []
    valid_nuclei_mask = np.zeros(zones.shape,'bool')
    # loop through each labeled spot.
    for region in measure.regionprops(labeled_int_signal_mask):
        region_mask = labeled_int_signal_mask == region.label
        x0, y0, x1, y1 = region.bbox
        # if x0 == dx0:
        #     break
        processed_bboxes.append(",".join([str(x) for x in [x0, y0, x1, y1]]))
        avg_zone_number = int(round(np.median(zones[region_mask]), ndigits=0))
        if avg_zone_number == 0:
            continue
        try:
            dapi_t = filters.threshold_otsu(dapi[x0:x1, y0:y1])
            print('DAPI threshold for cells detection: {}'.format(dapi_t))
        except ValueError:
            continue
        region_int_mask = int_signal_mask[x0:x1, y0:y1] & region.image
        region_dapi_mask = (dapi[x0:x1, y0:y1] > dapi_t) & region.image
        distance = ndi.distance_transform_edt(region_dapi_mask)
        local_maxi = feature.peak_local_max(
            distance, indices=False, min_distance=5
        )
        markers = measure.label(local_maxi, connectivity=2)
        labels = segmentation.watershed(
            -distance, markers, mask=region_dapi_mask)
        n_cells = 0
        for nuclei_mask_region in measure.regionprops(labels):
            if nuclei_mask_region.equivalent_diameter >= 5:
                n_cells += 1
                nx0, ny0, nx1, ny1 = nuclei_mask_region.bbox
                valid_nuclei_mask[x0+nx0:x0+nx1, y0+ny0:y0+ny1] = \
                    nuclei_mask_region.filled_image
        if n_cells != 0:
            # saving data
            clonal_sizes.append(n_cells)
            spot_sizes.append(region.equivalent_diameter)
            zone_lables.append(avg_zone_number)
            parent_bbox.append(",".join([str(x) for x in [x0, y0, x1, y1]]))
    spot_sizes_df = pd.DataFrame(
        {
            "clonal_size": clonal_sizes,
            "zone": zone_lables,
            "parent_bbox": parent_bbox,
            "spot_size_d": spot_sizes
        }
    )
    skipped_bbox = [
        x for x in processed_bboxes if x not in spot_sizes_df.parent_bbox.values
    ]
    return spot_sizes_df, skipped_bbox, valid_nuclei_mask
