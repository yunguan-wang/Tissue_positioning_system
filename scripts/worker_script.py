import argparse
from tps.find_zones import *
from tps.plotting import *
from tps.segmentation import *
import os
from skimage import io, filters, measure
import pandas as pd
import warnings
import matplotlib
import sys

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    matplotlib.use("Agg")
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
        default=False,
        action='store_true',
        help="If zonal spot sizes are calculated. This will only work for sparse signals.",
    )
    parser.add_argument(
        "-u",
        "--update",
        default=False,
        action='store_true',
        help="If true, will always overwrite existing results.",
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
    parser.add_argument(
        "-l",
        "--logging",
        default=False,
        action='store_true',
        help="Log file name, if empty, will print to command line. Otherwise, will make a log.txt file in the output.",
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
    tomato_cutoff = args.tomato_cutoff
    dapi_dilation_r = args.dapi_dilation_r
    logging = args.logging

    output_prefix = input_tif_fn.replace(".tif", "/")
    if output != "":
        output_prefix = os.path.join(output, output_prefix.split("/")[-2], "")
    output_mask_fn = output_prefix + "masks.tif"
    img = io.imread(input_tif_fn)
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)
    # make log file
    if logging:
        log_fn = output_prefix + "log.txt"
        log = open(log_fn, "w")
        sys.stdout = log
        sys.stderr = log
    print("Prosessing {}".format(input_tif_fn))
    print("Parameters: {}".format(args))
    # save an copy for reference
    _ = plt.figure(figsize=(16, 9))
    io.imshow(img)
    plt.savefig(output_prefix + "original_figure.pdf")
    plt.close()

    if os.path.exists(output_prefix + "zone int.csv") & (not update):
        print("Analysis already done, skip this job.")
    else:
        if os.path.exists(output_mask_fn) & (not update):
            print("Use existing masks")
            masks = io.imread(output_mask_fn)
            _, _, gs_ica = extract_gs_channel(img)
            # vessels = segmenting_vessels(
            #     img,
            #     dark_t=dapi_cutoff,
            #     dapi_channel=2,
            #     vessel_size_t=vessel_size_factor,
            #     dapi_dilation_r=dapi_dilation_r,
            # )
            # print("Merging neighboring vessel masks...")
            # vessels = measure.label(vessels, connectivity=2)
            # new_merged_mask, _ = merge_neighboring_vessels(vessels, max_dist=max_dist)
            # while not (new_merged_mask == vessels).all():
            #     vessels = new_merged_mask
            #     print("Continue merging neighboring masks...")
            #     new_merged_mask, _ = merge_neighboring_vessels(
            #         vessels, max_dist=max_dist
            #     )
            # vessels = new_merged_mask
        else:
            print("Segmentating using GS and DAPI")
            try:
                masks, gs_ica, vessels = segmenting_vessels_gs_assisted(
                    img,
                    vessel_size_t=vessel_size_factor,
                    max_dist=max_dist,
                    dark_t=dapi_cutoff,
                    dapi_dilation_r=dapi_dilation_r,
                )
            except:
                print(
                    "Default DAPI cutoff failed, try using 0.5 * Otsu threshold values."
                )
                dapi_cutoff = 0.5 * filters.threshold_otsu(img[:, :, 2])
                masks, gs_ica, vessels = segmenting_vessels_gs_assisted(
                    img,
                    vessel_size_t=vessel_size_factor,
                    max_dist=max_dist,
                    dark_t=dapi_cutoff,
                    dapi_dilation_r=dapi_dilation_r,
                )
            # Save masks
            io.imsave(output_mask_fn, masks.astype(np.uint8))
            # save gs_ica
            io.imshow(gs_ica)
            plt.savefig(output_prefix + "GS_ica.pdf")
            plt.close()

        # get CV PV classification
        cv_features = extract_features(
            masks, gs_ica, q1=gs_low, q2=gs_high, step=gs_step
        )
        print(cv_features.head(3))
        cv_labels, pv_labels = pv_classifier(cv_features.loc[:, "I0":], masks)
        cv_masks = masks * np.isin(masks, cv_labels).copy()
        pv_masks = masks * np.isin(masks, pv_labels).copy()
        # modify CV masks to shrink their borders
        gs_bool, _, _ = extract_gs_channel(img)
        gs_vessel = segmenting_vessels(
            (gs_bool + 0) + (masks == 0 + 0),
            dark_t=1,
            vessel_size_t=1,
            dapi_dilation_r=0,
        )
        masks = shrink_cv_masks(cv_masks, pv_masks, gs_vessel | vessels)

        # Updates masks
        cv_masks = masks * np.isin(masks, cv_labels).copy()
        pv_masks = masks * np.isin(masks, pv_labels).copy()
        cv_pv_masks = plot3channels(
            img[:, :, 2], cv_masks != 0, pv_masks != 0,
            fig_name = output_prefix + "Masks",
            return_array = True
        )
        io.imsave(output_prefix + 'cv_pv_masks.tif', cv_pv_masks.astype(np.uint8))
        # # save the masks to be used in tpsDeep
        # classified_masks = np.zeros((3, img.shape[0], img.shape[1]), bool)
        # classified_masks[0, :,:] = cv_masks
        # classified_masks[1, :,:] = pv_masks
        # classified_masks[2, :,:] = vessels!=0
        # mask_name = output_prefix + 'Classified_masks_' + output_prefix.split('/')[-2] + '.tif'
        # io.imsave(mask_name, classified_masks)

        # find lobules
        cv_masks = cv_masks.astype("uint8")
        _, lobules_sizes, lobule_edges = find_lobules(
            cv_masks, lobule_name=output_prefix.replace(".tif", "")
        )
        lobules_sizes.to_csv(output_prefix + "lobule_sizes.csv")
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
    if logging:
        log.close()
