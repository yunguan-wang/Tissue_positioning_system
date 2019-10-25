import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pylab import cm
import skimage as ski
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


def plot_zone_with_img(img, zones, fig_prefix=""):
    plot_zones = zones.copy()
    n_zones = np.unique(zones).shape[0] - 1
    plot_zones[plot_zones == -1] = n_zones + 2
    plot_zones[plot_zones == 255] = n_zones + 1
    plot_zones = plot_zones * 255 / (n_zones + 4)
    plt.imshow(img)
    plt.imshow(plot_zones, alpha=0.5)
    if fig_prefix != "":
        plt.savefig(fig_prefix + " zones with image.png", dpi=300)
        plt.close()
    plt.imshow(plot_zones)
    plt.savefig(fig_prefix + " zones only.png", dpi=300)
    plt.close()


def plot_zone_int(
    int_img,
    dapi_img,
    zone_mask,
    plot_type="box",
    savefig=False,
    marker_name="GLS2",
    prefix="",
):
    sns.set(style="white")
    zone_int = pd.DataFrame(columns=["zone"])
    dapi_cutoff = ski.filters.threshold_otsu(dapi_img)
    for zone in np.unique(zone_mask):
        if zone == 0:
            continue
        _zone_int_mask = (zone_mask == zone) & (dapi_img >= dapi_cutoff)
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
        plt.close()
    return zone_int


def plot_zone_int_probs(
    int_img,
    dapi_int,
    zone_mask,
    dapi_cutoff=20,
    savefig=False,
    plot_type="prob",
    marker_name="GLS2",
    prefix="",
):
    int_cutoff = ski.filters.threshold_otsu(int_img)
    if dapi_cutoff == "otsu":
        dapi_cutoff = ski.filters.threshold_otsu(dapi_int)
    int_signal_mask = int_img > int_cutoff
    zone_int_stats = pd.DataFrame(columns=["zone"])
    total_pos_int = (
        (zone_mask != 0) & int_signal_mask & (dapi_int > dapi_cutoff)
    ).sum()
    for zone in np.unique(zone_mask):
        if zone == 0:
            continue
        # pixels in zones
        _zone_px_mask = zone_mask == zone
        # valid pixels in zone where it is dapi positive
        _valid_zone_px_mask = _zone_px_mask & (dapi_int > dapi_cutoff)
        _num_total_px = _zone_px_mask.sum()
        _num_valid_px = _valid_zone_px_mask.sum()
        # pixels where is it signal positive and dapi positive
        _num_pos_px = int_signal_mask[_valid_zone_px_mask].sum()
        # weighted possibility of a pixel being signal positive
        # the weight is the pecentage of pixels in the zone being dapi positive
        _percent_pos = 100 * _num_pos_px / _num_total_px
        _zone_ints = pd.DataFrame(
            [_percent_pos], columns=["Possibility of observe positive signal"]
        )
        if zone == -1:
            _zone_ints["zone"] = "CV"
        elif zone == 255:
            _zone_ints["zone"] = "PV"
        else:
            _zone_ints["zone"] = "Z" + str(int(zone))
        _zone_ints["percent_valid"] = 100 * _num_valid_px / _num_total_px
        _zone_ints["percent_postive_in_zone"] = 100 * _num_pos_px / total_pos_int
        zone_int_stats = zone_int_stats.append(_zone_ints, ignore_index=True)
    zone_int_stats.loc[zone_int_stats.percent_valid < 10, "zone"] = (
        "*" + zone_int_stats.loc[zone_int_stats.percent_valid < 10, "zone"]
    )
    if plot_type == "probs":
        # smoothing the curving with polynomial fit
        poly = np.polyfit(zone_int_stats.index, zone_int_stats.iloc[:, 0], 3)
        poly_y = np.poly1d(poly)(zone_int_stats.index)
        # zone_int_stats["Possibility of observe positive signal"] = poly_y
        sns.lineplot(
            data=zone_int_stats,
            y="Possibility of observe positive signal",
            x="zone",
            sort=False,
        )
    else:
        poly = np.polyfit(zone_int_stats.index, zone_int_stats.iloc[:, 1], 3)
        poly_y = np.poly1d(poly)(zone_int_stats.index)
        # zone_int_stats["percent_postive_in_zone"] = poly_y
        sns.lineplot(
            data=zone_int_stats, y="percent_postive_in_zone", x="zone", sort=False
        )
    if prefix != "":
        plt.savefig(prefix + " signal intensity in zones.png", dpi=300)
        plt.close()
    return zone_int_stats
