#%%
# import argparse
import os
import sys
import logging
import argparse
from skimage import io, filters, measure
from tps.find_zones import *
from tps.plotting import *
from tps.segmentation import *

#%%
def filter_masks(masks):
    # iteratively merging masks until it cannot be merged.
    labeled_mask = measure.label(masks)
    mask_id, id_counts = np.unique(labeled_mask, return_counts=True)
    valid_ids = mask_id[id_counts>=150][1:]
    labeled_mask = labeled_mask * np.isin(labeled_mask, valid_ids)
    new_merged_mask, _ = merge_neighboring_vessels(
        labeled_mask, max_dist=20)
    while not (new_merged_mask == labeled_mask).all():
        labeled_mask = new_merged_mask
        print("Continue merging neighboring masks...")
        new_merged_mask, _ = merge_neighboring_vessels(labeled_mask, max_dist=20)
    return labeled_mask

def pool_masks(l_cv_masks, l_pv_masks):
    # merge two non-overlapping lebeled masks together.
    # ensure there is no overlap
    # set(np.unique(l_cv_masks))&set(np.unique(l_pv_masks)) == {0}
    masks = l_cv_masks + l_pv_masks
    output_masks = np.zeros(masks.shape, int)
    cv_labels = []
    pv_labels = []
    labeled_masks = measure.label(masks, connectivity = 2)
    for label, region in enumerate(measure.regionprops(labeled_masks)):
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

def worker_process(param_list):
    tif_fn, spot_size, tomato_cutoff, output_prefix, mask_fn = param_list
    if not os.path.exists(output_prefix):
        os.makedirs(output_prefix)
    img = io.imread(tif_fn)
    io.imsave(output_prefix+'/raw_image_scaled.tif', img)

    # setup logs
    log_fn = os.path.join(output_prefix, "log.txt")
    logging.basicConfig(
        filename=log_fn,
        format="%(asctime)s,%(levelname)s:::%(message)s",
        datefmt="%H:%M:%S",
        level="INFO",
    )
    stdout_logger = logging.getLogger("STDOUT")
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl
    stderr_logger = logging.getLogger("STDERR")
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    output_prefix += '/'
    masks = io.imread(mask_fn)
    fixed_masks = np.zeros(masks.shape)
    cv_masks = measure.label(masks[:,:,1]>0)
    pv_masks = measure.label(masks[:,:,2]>0)
    cv_masks = filter_masks(cv_masks) != 0
    pv_masks = filter_masks(pv_masks) != 0
    # pv_masks[pv_masks>0] = pv_masks[pv_masks>0] + cv_masks.max()
    masks, cv_labels, pv_labels, cv_masks, pv_masks = pool_masks(
        cv_masks, pv_masks)

    fixed_masks[:,:,1] = cv_masks != 0
    fixed_masks[:,:,2] = pv_masks != 0
    io.imsave(output_prefix+'/refined_masks.tif', fixed_masks)
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
    if not tomato_cutoff:
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
        if spot_sizes_df.shape[0] > 0:
            plot_spot_clonal_sizes(
                spot_sizes_df,
                absolute_number=False,
                figname=output_prefix + "spot_clonal_sizes.pdf",
            )
            spot_sizes_df.to_csv(output_prefix + "spot clonal sizes.csv")
    return

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass
#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Worker script for TPS deep-learning",
    )
    parser.add_argument(
        "input_img",
        type=str,
        help="Input TIF image to be zonated, with signal of interest at channel 0, \
                            GS at channel 1 and DAPI at channel 2",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="output folder of results, if not supplied, it will be that same as the input file name.",
    )
    parser.add_argument(
        "-s",
        "--spot_size",
        default=False,
        action='store_true',
        help="If zonal spot sizes are calculated. This will only work for sparse signals.",
    )
    parser.add_argument(
        "-tc",
        "--tomato_cutoff",
        default=False,
        type=float,
        help="Forced tomato cutoff, used in place of OTSU thresholding.",
    )

    # Parse all arguments
    # args = parser.parse_args(
    #     ['../dl_model/data/2w_Control_Cyp1a2-0301-f345-L1.tif'])
    args = parser.parse_args()
    input_tif_fn = args.input_img
    output = args.output
    spot_size = args.spot_size
    tomato_cutoff = args.tomato_cutoff
    
    # Config paths
    tps_path = os.path.abspath(__file__)
    infererence_path = tps_path.replace(
        "scripts/tps_worker.py", "dl_model/inference.py")
    model_path = tps_path.replace(
        "scripts/tps_worker.py", "dl_model/pretrained/tps_model.pt")
    if output is None:
        output = input_tif_fn.replace('.tif','/')
    if not os.path.exists(output):
        os.makedirs(output)
    mask_fn = input_tif_fn.replace('.tif','_mask.tif')
    log_fn = os.path.join(output, "log.txt")
    # setup log
    
    logging.basicConfig(
        filename=log_fn,
        format="%(asctime)s,%(levelname)s:::%(message)s",
        datefmt="%H:%M:%S",
        level="INFO",
    )
    stdout_logger = logging.getLogger("STDOUT")
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl
    stderr_logger = logging.getLogger("STDERR")
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl
    
    print('Segmenting CV and PV masks from {}...'.format(input_tif_fn))
    os.system(
        'python {} --input {} --rescale_factor 1.0 --device cpu --model {} --output {}'.format(
            infererence_path, input_tif_fn,model_path, mask_fn)
    )
    print('TPS zonal analysis...')
    worker_process([input_tif_fn, spot_size, tomato_cutoff, output, mask_fn])
    
    '''
    # Scripts used to process data included in paper.
    
    import pandas as pd
    from multiprocessing import Pool
    input_folder = '/endosome/archive/shared/zhu_wang/TPS_paper/Gozdeep_output'
    output_folder = '/endosome/work/InternalMedicine/s190548/TPS/tps_results'
    tifs = [x for x in os.listdir(input_folder) if '_mask.tif' not in x]
    design = pd.read_csv(
        '/endosome/work/InternalMedicine/s190548/TPS/design.txt', 
        sep='\t', index_col=0)
    for outfolder in design.folder.unique():
        if not os.path.exists(output_folder + '/' + outfolder):
            os.mkdir(output_folder + '/' + outfolder)
    params = []
    for tif in design.index:
        tif_fn = os.path.join(input_folder, tif)
        if not os.path.exists(tif_fn):
            continue
        spot_size,tomato_cutoff = design.loc[tif,['spot','tomato_cutoff']]
        tif_fn = os.path.join(input_folder, tif)
        mask_fn = tif_fn.replace('.tif','_mask.tif')
        spot_size = design.loc[tif,'spot']
        output_prefix = os.path.join(
            output_folder, design.loc[tif,'folder'], tif[:-4])
        params.append([tif_fn, spot_size, tomato_cutoff, output_prefix, mask_fn])
    P = Pool(16)
    P.map(worker_process, params)
    '''
