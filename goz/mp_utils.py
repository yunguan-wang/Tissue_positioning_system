from multiprocessing import Pool
from skimage import io, exposure
import numpy as np
import os
from goz.segmentation import *
from goz.plotting import plot3channels


def worker_segmentation(args):
    crop_coord, crop_img, crop_gs_ica, max_dist, dark_t, mask_prefix, job_id = args
    fn = mask_prefix + "_" + " ".join([str(x) for x in crop_coord]) + "_masks.tif"
    if os.path.exists(fn):
        print("Job {} is previously done, skipping this job.".format(job_id))
        return
    else:
        print("Processing job {}...".format(job_id))
    crop_img[crop_img == 0] = 1
    crop_img_norm = exposure.equalize_adapthist(crop_img)
    crop_img_norm = (crop_img_norm * 255).astype("uint8")
    masks, _, vessels = segmenting_vessels_gs_assisted(
        crop_img_norm, size_cutoff_factor=2, max_dist=max_dist, dark_t=dark_t, gs_ica=crop_gs_ica
    )
    img = np.zeros(masks.shape + (3,), "bool")
    img[:, :, 0] = masks
    img[:, :, 1] = vessels
    fn = mask_prefix + "_" + " ".join([str(x) for x in crop_coord]) + "_masks.tif"
    io.imsave(fn, img)
    plot3channels(img[:,:,2], img[:,:,0],img[:,:,1], fn[1:].replace('.tif',''))


def mp_segmentation(
    img, gs_ica, valid_crops, mask_fn_prefix, max_dist=10, dark_t=20, ntasks=8
):
    jobs_params = []
    for job_id, crop_coord in enumerate(valid_crops):
        top, bottom, left, right = crop_coord
        crop_img = img[top:bottom, left:right, :]
        crop_gs_ica = gs_ica[top:bottom, left:right]
        jobs_params.append(
            [
                crop_coord,
                crop_img,
                crop_gs_ica,
                max_dist,
                dark_t,
                mask_fn_prefix,
                job_id,
            ]
        )
    num_processors = 8
    p = Pool(processes=num_processors)
    _ = p.map(worker_segmentation, jobs_params)
    return
