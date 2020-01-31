import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pylab import cm
import skimage as ski
import numpy as np
import pandas as pd
import os

sns.set(style="white", rc={"axes.facecolor": "white", "figure.facecolor": "white"})


def plot3channels(c1, c2, c3, fig_name=None, return_array=False):
    # plot the lobules
    new_img = np.zeros((c1.shape[0], c1.shape[1], 3), dtype="uint8")
    for i, _array in enumerate([c1, c2, c3]):
        _array = _array + 0.0
        _array = (255 * _array / _array.max()).astype("uint8")
        new_img[:, :, i] = _array
    ski.io.imshow(new_img)
    if fig_name is not None:
        plt.savefig(fig_name + ".pdf", dpi=300, facecolor="w")
        plt.close()
    if return_array:
        return new_img


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


def plot_zone_with_img(img, zones, fig_prefix=None, tomato_channel=0, **kwargs):
    plot_zones = zones.copy()
    tomato = img[:, :, tomato_channel]
    tomato = tomato / 4
    tomato[0, 0] = 255
    vessel = np.zeros(zones.shape, "uint8")
    vessel[plot_zones == -1] = 0.5
    vessel[plot_zones == 255] = 2
    plot_zones[np.isin(plot_zones, [-1, 0, 255])] = 0
    plot_zones = 1 + plot_zones.max() - plot_zones
    plot_zones[plot_zones == plot_zones.max()] = 0
    return plot3channels(tomato, vessel, plot_zones, fig_prefix, **kwargs)


def plot_zones_only(zones, fig_prefix=None):
    plot_zones = zones.copy()
    # zone -1 is CV, and zone 255 is PV
    c1 = plot_zones == -1
    c2 = plot_zones == 255
    plot_zones[np.isin(plot_zones, [-1, 0, 255])] = 0
    plot3channels(plot_zones, c1, c2, fig_prefix)


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
    # quick hack for images where the tomato is too sparse
    if int_cutoff < 100:
        print("Tomato intensity threshold too low, override to 100!")
        int_cutoff = 100
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
            [_percent_pos], columns=["Possibility of observing positive signal"]
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
        # zone_int_stats["Possibility of observing positive signal"] = poly_y
        sns.lineplot(
            data=zone_int_stats,
            y="Possibility of observing positive signal",
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


def plot_pooled_zonal_data(
    plot_data,
    y_col="Possibility of observing positive signal",
    hue_col="Condition",
    hue_order=None,
    plot_diff_bar=True,
    plot_type="line",
    n_ticks=10,
    figname=None,
    forced_color=None,
):
    """Make zone intensity plot summerizing all image tiles. Optionally can be
    used to compare between conditions.

    Paremeters
    ========
    plot_data : pd.DataFrame
        a table with zonal data, must have a col'zone' with zone information and other columns
        for data. Indedices in the data carris no information.
        It is returned by "get_pooled_zone_int" function.
    (y_, hue_)col : str
        columns names for numberic data to be shown and hue color.
    plot_type : str
        type of plot, can be 'line' plot for zonal intensities and 'swarm'/'box' plot for sparsespot sizes.
    plot_diff_bar : bool
        if Tuue a bar graph representing the mean difference of each bin between conditionsis plotted.
    n_ticks : int
        number of ticks on x axis.
    forced_color : int
        hsl h value for color, for example, 240 is blue, and 0 is red.
    """
    # Drop CV
    plot_data = plot_data[plot_data.zone != "CV"]
    # sorting
    if not isinstance(plot_data.zone.dtypes, int):
        plot_data.zone = [x.replace("Z", "") for x in plot_data.zone.values]
    plot_data.zone = plot_data.zone.str.zfill(2)
    plot_data = plot_data.sort_values("zone")

    if len(plot_data) != 0:
        n_conditions = plot_data[hue_col].nunique()
        if hue_order is None:
            hue_order = sorted(plot_data[hue_col].unique())
        palette = sns.diverging_palette(
            240, 0, sep=80, n=n_conditions, s=75, l=50, center="dark"
        )
        if forced_color is not None:
            palette = sns.diverging_palette(
                forced_color, 0, sep=80, n=1, s=75, l=50, center="dark"
            )
        plot_data.rename(columns={"zone": "bin"}, inplace=True)
        if plot_type == "line":
            sns.lineplot(
                data=plot_data,
                x="bin",
                sort=False,
                y=y_col,
                hue=hue_col,
                hue_order=hue_order,
                palette=palette,
            )
            # Plot diff bars
            if (plot_diff_bar) & (n_conditions > 1):
                ref_cond = hue_order[0]
                target_cond = hue_order[1]
                ref = plot_data[plot_data[hue_col] == ref_cond].groupby("bin")
                tar = plot_data[plot_data[hue_col] == target_cond].groupby("bin")
                ref_mean = ref[y_col].mean()
                tar_mean = tar[y_col].mean()
                diff = tar_mean - ref_mean
                diff = diff[plot_data.bin.unique()]
                sns.barplot(x=diff.index, y=diff.values, color="grey", alpha=0.5)
        # alternative plots
        elif plot_type == "swarm":
            sns.swarmplot(
                x="bin",
                y=y_col,
                hue=hue_col,
                data=plot_data,
                hue_order=hue_order,
                palette=palette,
                alpha=0.7,
                size=3,
            )
        elif plot_type == "box":
            sns.boxplot(
                x="bin",
                y=y_col,
                hue=hue_col,
                data=plot_data,
                hue_order=hue_order,
                palette=palette,
            )
        # formatting ticks and tick labels.
        num_zones = plot_data.bin.nunique()
        # Starting with 1, getting rid of CV.
        ticks = np.linspace(0, num_zones - 1, n_ticks, dtype=int)
        tick_labels = plot_data.bin.unique()[ticks]
        _ = plt.xticks(ticks, tick_labels)
        _ = plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
        if figname is not None:
            plt.tight_layout()
            plt.savefig(figname, facecolor="w", dpi=300)
            plt.close()


def get_pooled_zonal_data(folders, markers, filename="zone int.csv"):
    """Prepare data for zone intensity plot.

    Paremeters
    ========
    folders : list
        a list of root folders where individual image tile and associated
        analysis resutls can be found.
    markers : list
        a list of markers to evaluate.
    filename : str
        filenames to be pooled in each named folder.

    Returns
    ========
    pooled_data : pd.DataFrame
        table of pooled zone intensity feature to be used in the plotting function.
    """
    if isinstance(folders, str):
        folders = [folders]
    if isinstance(markers, str):
        markers = [markers]
    pooled_data = pd.DataFrame()
    abs_path = os.getcwd()
    for folder in folders:
        for marker in markers:
            tif_files = sorted(
                [
                    x
                    for x in os.listdir(folder)
                    if (".tif" in x) & (marker.lower() in x.lower())
                ]
            )
            for img_fn in tif_files:
                output_prefix = img_fn.replace(".tif", "")
                _zonal_data_fn = os.path.join(abs_path, folder, output_prefix, filename)
                if not os.path.exists(_zonal_data_fn):
                    # Attempt to fix file name inconsistency caused by
                    # mis-capped gene names. Find matched folders by
                    # attempting lower cases.
                    folders_lower = [x.lower() for x in folders]
                    _lc_folder = os.path.join(abs_path, folder.lower())
                    if _lc_folder in folders_lower:
                        _mapped_folder = folders[folders_lower.index(_lc_folder)]
                        _zonal_data_fn = os.path.join(
                            abs_path, _mapped_folder, filename
                        )
                        print("Found match for {}: {}".format(marker, _zonal_data_fn))
                    else:
                        print("{} does not exist!".format(_zonal_data_fn))
                        continue
                _zonal_data = pd.read_csv(_zonal_data_fn, index_col=0)
                if filename == "zone int.csv":
                    _zonal_data.zone = [x.replace("*", "") for x in _zonal_data.zone]
                    _zonal_data.zone = _zonal_data.zone.replace("*CV", "CV")
                _zonal_data["Condition"] = " ".join([folder.split("/")[-1], marker])
                pooled_data = pooled_data.append(_zonal_data, sort=False)
    return pooled_data
