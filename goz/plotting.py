import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pylab import cm
from skimage import io, filters, morphology
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
    io.imshow(new_img)
    if fig_name is not None:
        if '.png' not in fig_name:
            plt.savefig(fig_name + ".pdf", dpi=300, facecolor="w")
            plt.close()
        else:
            plt.savefig(fig_name, dpi=300, facecolor="w")
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
        plt.savefig(prefix + "segmented_classfied.pdf", dpi=300)
        plt.close()


def plot_zone_with_img(img, zones, fig_prefix=None, tomato_channel=0, **kwargs):
    plot_zones = zones.copy()
    tomato = img[:, :, tomato_channel]
    tomato = tomato / 4
    tomato[0, 0] = 255
    vessel = np.zeros(zones.shape, "uint8")
    vessel[plot_zones == -1] = 2
    vessel[plot_zones == 255] = 0.5
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
    dapi_cutoff = filters.threshold_otsu(dapi_img)
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
            prefix + "{} zones marker intensity plot.pdf".format(str("")), dpi=300
        )
        plt.close()
    return zone_int


def plot_zone_int_probs(
    int_img,
    dapi_int,
    zone_crit,
    dapi_cutoff=20,
    savefig=False,
    plot_type="prob",
    marker_name="GLS2",
    tomato_cutoff = 0,
    prefix=""):
    int_cutoff = filters.threshold_otsu(int_img)
    # quick hack for images where the tomato is too sparse
    if int_cutoff < 100:
        print("Tomato intensity threshold too low, override to 100!")
        int_cutoff = 100
    # forced tomato cutoff value
    if tomato_cutoff != 0:
        int_cutoff = int(tomato_cutoff)
    if dapi_cutoff == "otsu":
        dapi_cutoff = filters.threshold_otsu(dapi_int)
    dapi_mask_exp = morphology.dilation(dapi_int> dapi_cutoff, morphology.disk(5))
    dapi_mask = dapi_int > dapi_cutoff
    int_signal_mask = int_img > int_cutoff
    # initialize the zone int dataframe
    n_zones = int((zone_crit[zone_crit<255]).max())
    print('number of zones : {}'.format(n_zones))
    zone_names = ['Z' + str(int(zone)) for zone in range(1,1+n_zones)]
    # zone_names = ['CV'] + zone_names + ['PV']
    zone_int_stats = pd.DataFrame(zone_names, columns=["zone"], index=np.arange(len(zone_names)))
    for idx,row in zone_int_stats.iterrows():
        zone = int(row.zone[1:])
        # pixels in zones
        _zone_px_mask = zone_crit == zone
        # Tomato area in zone
        _num_total_px = _zone_px_mask.sum()
        _num_pos_px = int_signal_mask[_zone_px_mask].sum()
        _percent_pos = 100 * _num_pos_px / _num_total_px
        # Tomato area as a function of total cellular area with expanded dapi mask
        _zonal_cell_area = (_zone_px_mask * dapi_mask_exp).sum()
        _zonal_marker_area = int_signal_mask[_zone_px_mask * dapi_mask_exp].sum()
        _percent_cellular_tomato = 100 * _zonal_marker_area / _zonal_cell_area
        # Tomato area as a function of total cellular area
        _zonal_cell_area = (_zone_px_mask * dapi_mask).sum()
        _zonal_marker_area = int_signal_mask[_zone_px_mask * dapi_mask].sum()
        _percent_cellular_tomato_no_exp = 100 * _zonal_marker_area / _zonal_cell_area
        # record data
        zone_int_stats.loc[idx, "percent of tomato area in zone"] = _percent_pos
        zone_int_stats.loc[idx, "percent of cellular tomato area in zone"] = _percent_cellular_tomato
        zone_int_stats.loc[idx, "percent of cellular tomato area in zone no exp"] = _percent_cellular_tomato_no_exp
    sns.lineplot(data=zone_int_stats, y="percent of tomato area in zone", x="zone", sort=False)
    if prefix != "":
        plt.savefig(prefix + " signal intensity in zones.pdf", dpi=300)
        plt.close()
    return zone_int_stats


def plot_pooled_zonal_data(
    plot_data,
    y_col="percent of tomato area in zone",
    hue_col="Condition",
    hue_order=None,
    plot_diff_bar=True,
    plot_type="line",
    n_ticks=10,
    figname=None,
    forced_color=None,
    forced_palette=None,
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
    plot_data = plot_data[plot_data.zone != "PV"]
    # sorting
    if not isinstance(plot_data.zone.dtypes, int):
        plot_data.zone = [x.replace("Z", "") for x in plot_data.zone.values]
    plot_data.zone = plot_data.zone.str.zfill(2)
    plot_data = plot_data.sort_values("zone")

    if len(plot_data) != 0:
        n_conditions = plot_data[hue_col].nunique()
        if hue_order is None:
            hue_order = sorted(plot_data[hue_col].unique())
        # setting pallete
        palette = sns.diverging_palette(
            240, 0, sep=80, n=n_conditions, s=75, l=50, center="dark"
        )
        if forced_color is not None:
            palette = sns.diverging_palette(
                forced_color, 0, sep=80, n=1, s=75, l=50, center="dark"
            )
        if forced_palette:
            palette = forced_palette
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
                [x for x in os.listdir(folder) 
                if (".tif" in x) & (marker.lower() in x.lower())])
            i = 0
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
                            abs_path, _mapped_folder, output_prefix,filename
                        )
                        if not os.path.exists(_zonal_data_fn):
                            print("{} does not exist!".format(_zonal_data_fn))
                            continue
                        print("Found match for {}: {}".format(_zonal_data_fn, _zonal_data_fn))
                    else:
                        print("{} does not exist!".format(_zonal_data_fn))
                        continue
                _zonal_data = pd.read_csv(_zonal_data_fn, index_col=0)
                if filename == "zone int.csv":
                    _zonal_data.zone = [x.replace("*", "") for x in _zonal_data.zone]
                    _zonal_data.zone = _zonal_data.zone.replace("*CV", "CV")
                _zonal_data["Condition"] = " ".join([folder.split("/")[-1], marker])
                _zonal_data["batch"] = 'batch' + str(i+1)
                pooled_data = pooled_data.append(_zonal_data, sort=False)
                i += 1
    return pooled_data

def spot_segmentation_diagnosis(img, spot_sizes_df,skipped_bboxes, fig_prefix='./'):
    '''
    Plot random 100 spot patches from both the good spots and bad spots.
    '''
    _, axes = plt.subplots(10,10, figsize=(40,40))
    axes = axes.ravel()
    spot_sizes_df = spot_sizes_df.set_index('parent_bbox').copy()
    n_subplots = np.min([100, spot_sizes_df.index.nunique()])
    random_patches = np.random.choice(spot_sizes_df.index.unique(),n_subplots, False)
    for i,bbox in enumerate(random_patches):
        px0,py0,px1,py1 = [int(x) for x in bbox.split(',')]
        axes[i].imshow(img[px0:px1,py0:py1,:])
        axes[i].set_title('+'.join(spot_sizes_df.clonal_size[[bbox]].astype(str)))
    plt.savefig(fig_prefix + 'valid_marker_spots.pdf')
    plt.close()

    _, axes = plt.subplots(10,10, figsize=(40,40))
    axes = axes.ravel()
    n_subplots = np.min([100, len(skipped_bboxes)])
    random_patches = np.random.choice(skipped_bboxes,n_subplots, False)
    for i,bbox in enumerate(random_patches):
        px0,py0,px1,py1 = [int(x) for x in bbox.split(',')]
        axes[i].imshow(img[px0:px1,py0:py1,:])
    plt.savefig(fig_prefix+'invalid_marker_spots.pdf')
    plt.close()

def plot_spot_clonal_sizes(
    spot_sizes_df,
    bins = [0,1,4,7,99],
    figname=None,
    absolute_number = True,
    ylab = '% of clone sizes in zone'):
    # bin breaks are in (,] format.
    # parsing bins and labels
    labels = []
    for i,bin_break in enumerate(bins[:-1]):
        bin_values = list(range(bin_break+1,bins[i+1]+1))
        if len(bin_values) == 1:
            labels.append(bin_values[0])
        elif bin_break == bins[-2]:
            labels.append('>=' + str(bin_break+1))
        else:
            labels.append('{}-{}'.format(bin_values[0],bin_values[-1]))

    plot_data = spot_sizes_df.copy()
    plot_data['clonal_sizes'] = pd.cut(
        plot_data.clonal_size, bins,labels=labels)
    plot_data = plot_data.groupby(['zone','clonal_sizes'])['clonal_size'].count()
    plot_data = plot_data.unstack('clonal_sizes').fillna(0)
    if not absolute_number:
        plot_data = plot_data.apply(lambda x: x/x.sum(),axis=1)

    plot_data.plot(kind='bar', stacked=True)
    plt.xlabel('GoZ zones')
    plt.xticks(rotation=0)
    plt.ylabel(ylab)
    plt.legend(bbox_to_anchor=(1,0.5),loc='center left')
    plt.tight_layout()
    if figname is not None:
        plt.savefig(figname)
        plt.close()