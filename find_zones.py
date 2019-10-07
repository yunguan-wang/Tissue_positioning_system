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


def get_distance_projection(masks, gs_labels, non_gs_labels):
    """
    Improved distance calculation algorithm aimed to get a distance measure that are
    linear to the observed distance to cv and pv.
    """
    coords_pixel = np.array(
        np.meshgrid(np.arange(masks.shape[0]), np.arange(masks.shape[1]), indexing="ij")
    )
    _, coords_cv = ndi.morphology.distance_transform_edt(
        ~np.isin(masks, gs_labels), return_indices=True
    )
    _, coords_pv = ndi.morphology.distance_transform_edt(
        ~np.isin(masks, non_gs_labels), return_indices=True
    )
    vector_cv = coords_pixel - coords_cv
    vector_pv = coords_pv - coords_pixel
    vector_pv_to_cv = coords_pv - coords_cv
    vector_dot = np.einsum("ijk,ijk->jk", vector_cv, vector_pv_to_cv)
    vector_pv_to_cv_norm = np.linalg.norm(vector_pv_to_cv, axis=0)
    projection = vector_dot / vector_pv_to_cv_norm / vector_pv_to_cv_norm
    vector_cv_norm = np.linalg.norm(vector_cv, axis=0)
    vector_pv_norm = np.linalg.norm(vector_pv, axis=0)
    cosine_vectors = (
        np.einsum("ijk,ijk->jk", vector_cv, vector_pv) / vector_cv_norm / vector_pv_norm
    )
    projection[cosine_vectors < 0] = -1
    return projection


def fill_hollow_masks(hollow_labeled_masks):
    filled_labeld_mask = hollow_labeled_masks
    for region in measure.regionprops(hollow_labeled_masks):
        x0, y0, x1, y1 = region.bbox
        filled_labeld_mask[x0:x1, y0:y1] |= region.filled_image * region.label
    return filled_labeld_mask


def dist_to_nn_masks(labeled_mask, target_labels, dist=None, nn=3):
    if dist is None:
        filled_labeld_mask = fill_hollow_masks(labeled_mask)
        dist = [
            ndi.distance_transform_edt(filled_labeld_mask != x) for x in target_labels
        ]
        dist = np.array(dist)
        dist = np.sort(dist, axis=0)
    # get mean distance to nearest n masks
    mean_dist = np.mean(dist[:nn, :, :], axis=0)
    return dist, mean_dist


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


def find_orphans(masks, gs_labels, non_gs_labels):
    dist_to_pv = ndi.morphology.distance_transform_edt(~np.isin(masks, non_gs_labels))
    dist_to_cv = ndi.morphology.distance_transform_edt(~np.isin(masks, gs_labels))
    orphans = (dist_to_pv > 350) | (dist_to_cv > 350)
    return orphans
