import pandas as pd
import skimage as ski
from skimage import io
import os
from segmentation import *
from plotting import *
from find_zones import *

liver1 = "T:/images/Gls2-1.tif"
liver2 = "T:/images/tert-1.tif"
img = io.imread(liver1)
img = np.array(img, dtype=np.uint8)

gs_ica = extract_gs_channel(img)
io.imshow(gs_ica)
masks = segmenting_vessels_gs_assisted(img, vessel_size_t=2)
gs_features = extract_features(masks, img[:, :, 1], q1=0.75, q2=0.99, step=0.1)
gs_labels, non_gs_labels = pv_classifier(gs_features.loc[:, "I0":], masks)
plot_pv_cv(masks, gs_labels, img)
