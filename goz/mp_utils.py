from multiprocessing import Process
import skimage as ski
import numpy as np
from goz.segmentation import *


def worker_segmentation(
    crop_coord, crop_img, crop_gs_ica, min_dist=10, dark_t=20, mask_prefix=""
):
    crop_img_norm = ski.exposure.equalize_adapthist(crop_img)
    crop_img_norm = (crop_img_norm * 255).astype("uint8")
    masks, _, vessels = segmenting_vessels_gs_assisted(
        crop_img_norm, min_dist=min_dist, dark_t=dark_t, gs_ica=crop_gs_ica
    )
    fn = mask_prefix + "_" + " ".join([str(x) for x in crop_coord]) + "_masks.png"
    ski.io.imsave(fn, masks)


def mp_segmentation(img, gs_ica, valid_crops, min_dist=10, dark_t=20, ntasks=8):
    n_batches = int(np.ceil(len(valid_crops) / ntasks))
    for j in range(n_batches):
        for _, crop_coord in enumerate(valid_crops):
            top, bottom, left, right = crop_coord
            crop_img = img[top:bottom, left:right, :]
            crop_gs_ica = gs_ica[top:bottom, left:right]
            p = Process(
                target=worker_segmentation,
                args=(crop_coord, crop_img, crop_gs_ica, min_dist, dark_t, "../Tests/"),
            )
            p.start()
            print(p.pid)
            p.join()
        print("Batch {} finished".format(j + 1))
    return
