import pandas as pd
from skimage import io, color, measure
from scipy import ndimage
# from scipy.signal import find_peaks,peak_widths
import numpy as np
import os
from goz.segmentation import *
from goz.plotting import *
from goz.find_zones import *
from goz.mp_utils import *
from goz.large_image_processing import *
import argparse
import warnings
import matplotlib
import sys

warnings.filterwarnings("ignore")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "False"):
        return True
    elif v.lower() in ("no", "false", "f", "True"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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
        default=0.1,
        help="The interval of percentage in the GS intensity features.",
    )
    parser.add_argument(
        "-s",
        "--spot_size",
        type=str2bool,
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
    parser.add_argument(
        "-tc",
        "--tomato_cutoff",
        nargs="?",
        type=int,
        default=0,
        help="Forced tomato cutoff, used in place of OTSU thresholding.",
    )
    parser.add_argument(
        "-dr",
        "--dapi_dilation_r",
        nargs="?",
        type=int,
        default=0,
        help="Dilation radius for dapi, useful in handle damage tissue image where cell death is prevalent.",
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
    tomato_cutoff = args.tomato_cutoff
    dapi_dilation_r = args.dapi_dilation_r

    if output == "":
        output_prefix = input_tif_fn.split(".")[0] + "/"
    else:
        output_prefix = output
    if not os.path.exists(output_prefix):
        os.makedirs(output_prefix)
    
    # make log file
    log_fn = output_prefix + "log"
    log = open(log_fn, "a")
    sys.stdout = log
    sys.stderr = log
    print("Prosessing {}".format(input_tif_fn))
    print("Parameters: {}".format(args))
    img = io.imread(input_tif_fn)
    dapi = img[:,:,2]

    # save an copy for reference
    _ = plt.figure(figsize=(16, 9))
    io.imshow(img)
    plt.savefig(output_prefix + "original_figure.pdf")
    plt.close()

    # # find the threshold for boundary discovery based on peak finding on the 
    # # histogram from channel intensity maxima.

    # # Use max intensity of each channel as criteria
    # # ignore 0 here
    # img_hist,bins = np.histogram(
    #     dapi.flatten(), bins=128, density=True, range=(1,255))
    # peaks, peaks_params = find_peaks(
    #     img_hist,height=.005, distance=5, width=2, rel_height=.9)
    # # dignostic scripts outputing the the peaks.
    # p_widths = peak_widths(img_hist,peaks,rel_height=.90)
    # plt.plot(img_hist)
    # plt.plot(peaks, img_hist[peaks], "x")
    # plt.hlines(*p_widths[1:], color="C3")
    # plt.savefig(output_prefix + 'DAPI intensity distribution peaks.pdf')

    # # Two boundries are defined, both on the left side and right side of the 
    # # first peak. Normally should use the left side one, however, sometimes the 
    # # image have artifacts near the boundary, cause another peak to form before 
    # # the actual peak fo the signal, thus in this case the right side boundry 
    # # are used.
 
    boundry_masks = find_boundry(dapi)

    img = img*boundry_masks[:,:,None]
    # save an copy for eroded image 
    _ = plt.figure(figsize=(16, 9))
    io.imshow(img)
    plt.savefig(output_prefix + "eroded_original_figure.pdf")
    plt.close()

    vessel_size_l = 2 * width * height / 10000
    # find valid image crops
    valid_crops = find_valid_crops(img[:, :, 2])
    plot_image_crops(img[:,:,2],pd.DataFrame(valid_crops),padding,output_prefix)
    # extract gs channel
    # Histogram equlication of GS channel
    gs_ica, _, raw_gs_ica = extract_gs_channel(img)

    # save gs_ica
    _ = plt.figure(figsize=(16,9))
    io.imshow(gs_ica)
    plt.savefig(output_prefix + "GS_ica.pdf")
    plt.close()

    # multiprocessing each image crop
    mp_segmentation(
        img,
        gs_ica,
        valid_crops,
        output_prefix,
        max_dist=max_dist,
        dark_t=dapi_cutoff,
        ntasks=ntasks,
        dapi_dilation_r=dapi_dilation_r,
    )

    crop_mask_files = [
        os.path.abspath(output_prefix + x)
        for x in os.listdir(output_prefix)
        if x[0] == "_"
    ]
    # pool and pruning cropped image masks.
    overall_masks, vessels = pool_masks_from_crops(
        img, crop_mask_files, boundry_masks, padding=padding
    )
    #! vessel masks is not pruned!!!
    good_masks = mask_pruning(overall_masks, vessel_size_l)
    vessels = vessels * good_masks
    # CV, PV classification
    cv_features = extract_features(
        good_masks, raw_gs_ica, q1=gs_low, q2=gs_high, step=gs_step
    )
    cv_labels, pv_labels = pv_classifier(
        cv_features.loc[:, "I0":], good_masks, max_cv_pv_ratio=1.5)
    # modify CV masks to shrink their borders
    cv_masks = good_masks * np.isin(good_masks, cv_labels).copy()
    pv_masks = good_masks * np.isin(good_masks, pv_labels).copy()
    good_masks = shrink_cv_masks(cv_masks, pv_masks, vessels)
    cv_masks = good_masks * np.isin(good_masks, cv_labels).copy()
    pv_masks = good_masks * np.isin(good_masks, pv_labels).copy()
    plot3channels(
        img[:, :, 2], cv_masks != 0, pv_masks != 0, fig_name=output_prefix + "Masks"
    )

    # Defining zones
    # Calculate distance projections
    #! orphan cut off set at 550
    zone_crit = calculate_zone_crit(cv_masks, pv_masks, tolerance=550)

    # modify zone_crit with valid image border
    zone_crit = zone_crit * boundry_masks
    processe_img_mask = np.zeros(img.shape[:2],'bool')
    for crop in valid_crops:
        x0,x1,y0,y1 = crop
        x0 = x0 + padding
        y0 = y0 + padding
        x1 = x1 - padding
        y1 = y1 - padding
        processe_img_mask[x0:x1,y0:y1] = 1
    zone_crit = zone_crit * processe_img_mask
    # Calculate zones
    zones = create_zones(
        good_masks,
        zone_crit,
        cv_labels,
        pv_labels,
        zone_break_type="equal_quantile",
        num_zones=24,
    )

    # Plot zones with image
    plot_zone_with_img(img, zones, fig_prefix=output_prefix + "zones with marker")
    plot_zones_only(zones, fig_prefix=output_prefix + "zones only")

    # Calculate zonal reporter expression levels.
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

    # Calculate zonal spot sizes.
    if spot_size:
        spot_sizes_df, skipped_boxes,valid_nuclei_masks = calculate_clonal_size(
            img, zones)
        spot_segmentation_diagnosis(
            img, spot_sizes_df, skipped_boxes,valid_nuclei_masks, 
            fig_prefix=output_prefix
        )
        spot_sizes_df.to_csv(output_prefix + "spot clonal sizes.csv")
        plot_spot_clonal_sizes(
        spot_sizes_df,
        absolute_number=False,
        figname=output_prefix + "spot_clonal_sizes.pdf",
        )
    log.close()
