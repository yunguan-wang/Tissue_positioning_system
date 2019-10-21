import pandas as pd
import skimage as ski
import numpy as np
from skimage import io
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale, minmax_scale
import skimage.measure as measure
import skimage.feature as feature
import skimage.filters as filters
import skimage.segmentation as segmentation
import skimage.morphology as morphology
import scipy.ndimage as ndi


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
    """
    Get average distance per pixel to nearest n masks.
    """
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


def make_zone_masks(dist_ratio, n_zones, method="division"):
    zone_mask = np.zeros(dist_ratio.shape)
    if method == "histogram":
        for zone in range(n_zones):
            zone_mask[(dist_ratio > bins[zone]) & (dist_ratio < bins[zone + 1])] = int(
                zone + 1
            )
    elif method == "division":
        interval = np.percentile(dist_ratio, 99) / n_zones
        for i in range(n_zones):
            zone_mask[
                (dist_ratio > i * interval) & (dist_ratio < (1 + i) * interval)
            ] = int((i + 1))
        zone_mask[dist_ratio >= np.percentile(dist_ratio, 99)] = int(i + 2)
    return zone_mask


def find_orphans(masks, cv_labels, pv_labels, orphan_crit=400):
    dist_to_pv = ndi.morphology.distance_transform_edt(~np.isin(masks, pv_labels))
    dist_to_cv = ndi.morphology.distance_transform_edt(~np.isin(masks, cv_labels))
    orphans = (dist_to_pv > orphan_crit) | (dist_to_cv > orphan_crit)
    return orphans


def create_zones(masks, zone_crit, cv_labels, pv_labels, zone_breaks=None, num_zones=5):
    # CV are labeled as -1
    # PV are labeled as -2
    zones = np.zeros(masks.shape[:2])
    if zone_breaks is None:
        for i in range(num_zones):
            t0 = i / num_zones
            t1 = (i + 1) / num_zones
            if i == num_zones - 1:
                zones[zone_crit > t0] = i + 1
            else:
                zones[(zone_crit > t0) & (zone_crit <= t1)] = i + 1
    else:
        for i, zone_break in enumerate(zone_breaks[:-1]):
            zones[(zone_crit > zone_break) & (zone_crit <= zone_breaks[i + 1])] = i + 1
    zones[np.isin(masks, cv_labels)] = -1
    zones[np.isin(masks, pv_labels)] = 255
    return zones
