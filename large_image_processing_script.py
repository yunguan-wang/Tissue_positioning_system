import pandas as pd
from skimage import io
import skimage as ski
import numpy as np
import os
from goz.segmentation import *
from goz.plotting import *
from goz.find_zones import *
from goz.mp_utils import *
from goz.large_image_processing import *
from multiprocessing import Pool
import argparse
import warnings
import matplotlib

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    matplotlib.use("Agg")
    plt = matplotlib.pyplot
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Worker script for Autozone",
    )
    parser.add_argument(
        "input_img",
        type=str,
        help="Absolute Input TIF image to be zonated, with signal of interest at channel 0, \
                            GS at channel 1 and DAPI at channel 2",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        nargs="?",
        default="",
        help="output folder of results, if not supplied, it will be that same as the input file name.",
    )
    parser.add_argument(
        "-v",
        "--vessel_size_factor",
        type=int,
        nargs="?",
        default=2,
        help="Vessel size threshold as x/10000 fold of image size",
    )
    parser.add_argument(
        "-d",
        "--maximal_neighbor_distance",
        type=int,
        nargs="?",
        default=20,
        help="maximal pixel distance between two neighboring masks to be considered as two separate masks.",
    )
    parser.add_argument(
        "-c",
        "--dapi_cutoff",
        type=int,
        nargs="?",
        default=20,
        help="Dapi cutoff value for hard thresholding.",
    )
    parser.add_argument(
        "-gl",
        "--gs_lower_limit",
        type=float,
        nargs="?",
        default=0.25,
        help="The lower percentatge limit of GS signal intensity within a mask, which is used in classify CV from PV",
    )
    parser.add_argument(
        "-gh",
        "--gs_higher_limit",
        type=float,
        nargs="?",
        default=0.75,
        help="The higher percentatge limit of GS signal intensity within a mask, which is used in classify CV from PV",
    )
    parser.add_argument(
        "-gs",
        "--gs_step",
        type=float,
        nargs="?",
        default=0.05,
        help="The interval of percentage in the GS intensity features.",
    )
    parser.add_argument(
        "-s",
        "--spot_size",
        type=bool,
        nargs="?",
        default=False,
        help="If zonal spot sizes are calculated. This will only work for sparse signals.",
    )
    parser.add_argument(
        "-u",
        "--update",
        type=bool,
        nargs="?",
        default=False,
        help="Check for existing analysis results, if exist, skip the job.",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        nargs="?",
        default=3500,
        help="width of each image crop.",
    )
    parser.add_argument(
        "-ht",
        "--height",
        type=int,
        nargs="?",
        default=1500,
        help="height of each image crop.",
    )
    parser.add_argument(
        "-p",
        "--padding",
        type=int,
        nargs="?",
        default=250,
        help="padding around each image crop.",
    )
    parser.add_argument(
        "-n",
        "--ntasks",
        type=int,
        nargs="?",
        default=8,
        help="number of image crop segmentation tasks",
    )
    # Parse all arguments
    args = parser.parse_args()
    input_tif_fn = args.input_img
    output = args.output
    vessel_size_factor = args.vessel_size_factor
    max_dist = args.maximal_neighbor_distance
    gs_low = args.gs_lower_limit
    gs_high = args.gs_higher_limit
    gs_step = args.gs_step
    update = args.update
    dapi_cutoff = args.dapi_cutoff
    spot_size = args.spot_size
    width = args.width
    height = args.height
    padding = args.padding
    ntasks = args.ntasks

    if output == "":
        output_prefix = input_tif_fn.split(".")[0] + "/"
    else:
        output_prefix = output
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)

    img = io.imread(input_tif_fn)
    vessel_size_l = 4 * width * height / 10000
    # find valid image crops
    valid_crops = find_valid_crops(img[:, :, 2])
    # extract gs channel
    gs_ica, _, _ = extract_gs_channel(img)
    # multiprocessing each image crop
    mp_segmentation(
        img,
        gs_ica,
        valid_crops,
        output_prefix,
        max_dist=max_dist,
        dark_t=dapi_cutoff,
        ntasks=ntasks,
    )

    crop_mask_files = [
        os.path.abspath(output_prefix + x)
        for x in os.listdir(output_prefix)
        if x[0] == "_"
    ]
    # pool and pruning cropped image masks.
    overall_masks, vessels = pool_masks_from_crops(
        img, crop_mask_files, padding=padding
    )
    good_masks = mask_pruning(overall_masks, vessel_size_l)

    # CV, PV classification
    cv_features = extract_features(
        good_masks, gs_ica, q1=gs_low, q2=gs_high, step=gs_step
    )
    cv_labels, pv_labels = pv_classifier(cv_features.loc[:, "I0":], good_masks)
    # modify CV masks to shrink their borders
    pv_masks = good_masks * np.isin(good_masks, pv_labels).copy()
    final_masks = shrink_cv_masks(good_masks, pv_masks, vessels)
    plot_pv_cv(final_masks, cv_labels, img, prefix=output_prefix)

    # Defining zones
    coords_pixel, coords_cv, coords_pv = find_pv_cv_coords(
        final_masks, cv_labels, pv_labels
    )
    orphans = find_orphans(final_masks, cv_labels, pv_labels, orphan_crit=1000)
    zone_crit = get_distance_projection(
        coords_pixel, coords_cv, coords_pv, cosine_filter=True
    )
    zone_crit[final_masks != 0] = -1
    zone_crit[(zone_crit > 1) | (zone_crit < 0)] = -1
    zone_crit[orphans] = -1

    # Calculate zones
    zones = create_zones(final_masks, zone_crit, cv_labels, pv_labels, num_zones=9)

    # Plot zones with image
    plot_zone_with_img(img, zones, fig_prefix=output_prefix + "Marker")

    # Calculate zonal spot sizes.
    if spot_size:
        _ = get_zonal_spot_sizes(img[:, :, 0], zones, output_prefix)
    # Calculate zonal reporter expression levels.
    zone_int = plot_zone_int_probs(
        img[:, :, 0],
        img[:, :, 2],
        zones,
        dapi_cutoff="otsu",
        plot_type="probs",
        prefix=output_prefix + "Marker",
    )
    zone_int.to_csv(output_prefix + "zone int.csv")

