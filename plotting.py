import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pylab import cm
import numpy as np
import pandas as pd


def plot_pv_cv(labeled_mask, gs_labels, img, prefix=""):
    new_mask = np.zeros(labeled_mask.shape)
    for _label in gs_labels:
        _crit = labeled_mask == _label
        _mask = (labeled_mask != 0) * _crit
        new_mask += _mask.astype(int)
    plt.figure(figsize=(16, 9))
    plt.imshow(img)
    plt.imshow(new_mask, cmap="Greys", alpha=0.7)
    plt.imshow((labeled_mask != 0 - new_mask), alpha=0.5)
    if prefix != "":
        plt.savefig(prefix + "segmented_classfied.png", dpi=300)
        plt.close()


def plot_zone_int(img, dapi, zone_mask, savefig=False, prefix=""):
    zone_means = []
    zone_std = []
    zone_list = []
    for zone in np.unique(zone_mask):
        if zone == 0:
            continue
        zone_list.append("Z" + str(zone))
        _zone_ints = img[(zone_mask == zone) & (dapi > 20)]
        _mean = _zone_ints.mean()
        _std = _zone_ints.std()
        zone_means.append(_mean)
        zone_std.append(_std)
    plt.bar(zone_list, zone_means, yerr=zone_std)
    if savefig:
        plt.savefig(
            prefix + "{} zones marker intensity plot.png".format(str("")), dpi=300
        )


def plot_zone_with_img(img, zone_mask, savefig=False, prefix=""):
    n_zones = np.unique(zone_mask).shape[0]
    plt.imshow(img)
    cmap = cm.get_cmap("PiYG", n_zones)
    plt.imshow(zone_mask, cmap=cmap, alpha=0.8)
    plt.colorbar()
    if savefig:
        plt.savefig(
            prefix + "{} zones marker intensity plot.png".format(str(n_zones)), dpi=300
        )


def plot_zone_int(
    int_img,
    dapi_img,
    zone_mask,
    plot_type="box",
    orphan_mask=None,
    savefig=False,
    marker_name="GLS2",
    prefix="",
):
    sns.set(style="white")
    zone_int = pd.DataFrame(columns=["zone"])
    for zone in np.unique(zone_mask):
        if zone == 0:
            continue
        _zone_int_mask = (zone_mask == zone) & (dapi_img >= 50)
        if orphan_mask is not None:
            _zone_int_mask = _zone_int_mask & (orphan_mask == False)
        _zone_ints = int_img[_zone_int_mask]
        _zone_ints = pd.DataFrame(_zone_ints, columns=["intensity"])
        _zone_ints["zone"] = "Z" + str(int(zone))
        zone_int = zone_int.append(_zone_ints, ignore_index=True)
    if plot_type == "box":
        sns.boxplot(x="zone", y="intensity", data=zone_int)
    elif plot_type == "violin":
        sns.violinplot(x="zone", y="intensity", data=zone_int)
    plt.title("{} intensity in different zones".format(marker_name))
    if savefig:
        plt.savefig(
            prefix + "{} zones marker intensity plot.png".format(str("")), dpi=300
        )
    return zone_int
