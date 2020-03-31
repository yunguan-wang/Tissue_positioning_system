import pandas as pd
import skimage as ski
from skimage import io
import os
from goz.segmentation import *
from goz.plotting import *
from goz.find_zones import *
from scipy.stats import ttest_ind, ttest_ind_from_stats
from itertools import product

'''
Scripts for making most of the figures in the paper
'''

def binning_clonal_sizes(pooled_data, bins = [0,2,5,7,99]):
    binned_data = pooled_data.copy()
    labels = []
    for i,bin_break in enumerate(bins[:-1]):
        bin_values = list(range(bin_break+1,bins[i+1]+1))
        if len(bin_values) == 1:
            labels.append(bin_values[0])
        elif bin_break == bins[-2]:
            labels.append('>=' + str(bin_break+1))
        else:
            labels.append('{}-{}'.format(bin_values[0],bin_values[-1]))
    binned_data['clonal_sizes'] = pd.cut(binned_data.clonal_size, bins,labels=labels)
    binned_data = binned_data.drop(['parent_bbox','clonal_size'],axis=1)
    binned_data['filler'] = 1
    return binned_data

def plot_spot_clonal_sizes(
    plot_data,
    figname=None,
    absolute_number = True,
    ylab = '% of clone sizes in zone',
    title='',
    ax=None,
    legend=False):
    # bin breaks are in (,] format.
    # parsing bins and labels
    plot_data = plot_data.copy()
    plot_data = plot_data.groupby(['zone','clonal_sizes'])['filler'].sum()
    plot_data = plot_data.unstack('clonal_sizes')
    if not absolute_number:
        plot_data = plot_data.apply(lambda x: x/x.sum(),axis=1)
    g = plot_data.plot(kind='bar', stacked=True,ax=ax)
    g.set_xlabel('GoZ zones')
    g.set_xticks([])
    g.set_ylabel(ylab)
    if legend == False:
        g.legend([])
    else:
        g.legend(bbox_to_anchor=(1,0.5), loc = 'center left')
    g.set_title(title)
    if figname is not None:
        plt.tight_layout()
        plt.savefig(figname)
        plt.close()

os.chdir("/home2/s190548/work_zhu/images/New Tif")
_=sns.plotting_context('paper', font_scale = 2)
cond_folders = [
    '/home2/s190548/work_zhu/images/New Tif/Normal  30%',
    '/home2/s190548/work_zhu/images/New Tif/6 months 30%']

markers = []
for cond_folder in cond_folders:
    markers += [x.split("-")[0] for x in os.listdir(cond_folder) if "tif" in x]
markers = list(set(markers))

############################################################################
# Figure 3a

cond_folders = ['/home2/s190548/work_zhu/images/New Tif/Normal  30%']
markers = []
for cond_folder in cond_folders:
    markers += [x.split("-")[0] for x in os.listdir(cond_folder) if "tif" in x]
markers = list(set(markers))
plot_data_t0 = get_pooled_zonal_data(cond_folders, markers)
plot_data_t0.zone = [x.replace("Z", "") for x in plot_data_t0.zone.values]
plot_data_t0.zone = plot_data_t0.zone.str.zfill(2)
plot_data_t0 = plot_data_t0.sort_values("zone")
plot_data_t0.Condition = plot_data_t0.Condition.apply(lambda x: x.split(' ')[-1].upper())
plot_data_t0 = plot_data_t0.sort_values('Condition')
plot_data_t0 = plot_data_t0.rename({'Condition':'Marker'}, axis=1)
# plot_data_t0 = plot_data_t0[~plot_data_t0.zone.isin(['01','24'])]
pan = ['Apoc4','Pklr']
zone1 = ['Arg11', 'Arg12','Gls2']
perip = ['Sox9','Krt19']
zone3 = ['Gs','Cyp1a2','Oat', 'Axin2']
zone2 = ['Hamp2','Mup3','Tert']
sns.set(style='white',font_scale=2.5)
fig, axes = plt.subplots(2,3, figsize=(20,10))
axes = axes.ravel()
i = 0
for marker_list, list_name in zip(
    [pan,zone1,perip,zone3,zone2],
    ['Pan zones','Zone 1 centric', 'Periportal', 'Zone 3 centric','Zone 2 and sparce']):
    marker_list = [x.upper() for x in marker_list]
    _plot_data = plot_data_t0[plot_data_t0.Marker.isin(marker_list)]
    # g = sns.FacetGrid(_plot_data,col='Condition',col_wrap=4, sharey=False)
    # g = g.map(sns.lineplot, 'zone','percent of cellular tomato area in zone no exp')
    g = sns.lineplot(
        x='zone', y='percent of cellular tomato area in zone no exp', hue='Marker',
        data = _plot_data, ax=axes[i])
    g.axes.set_xlabel('Goz zones')
    g.axes.set_ylabel('')
    g.axes.set_xticklabels([])
    g.axes.set_title(list_name)
    g.axes.legend(bbox_to_anchor = (1,.5), loc='center left', fontsize=20)
    i+=1
fig.delaxes(axes[i])
plt.tight_layout()
plt.savefig('/home2/s190548/work_zhu/images/New Tif/Figure3a.pdf')
plt.close()


#############################################################################
# Figure 3b

cond_folders = ['/home2/s190548/work_zhu/images/New Tif/6 months 30%']
markers = []
for cond_folder in cond_folders:
    markers += [x.split("-")[0] for x in os.listdir(cond_folder) if "tif" in x]
markers = list(set(markers))
plot_data_t6m = get_pooled_zonal_data(cond_folders, markers)
plot_data_t6m.zone = [x.replace("Z", "") for x in plot_data_t6m.zone.values]
plot_data_t6m.zone = plot_data_t6m.zone.str.zfill(2)
plot_data_t6m = plot_data_t6m.sort_values("zone")
plot_data_t6m.Condition = plot_data_t6m.Condition.apply(lambda x: x.split(' ')[-1].upper())
plot_data_t6m = plot_data_t6m.sort_values('Condition')
plot_data_t6m = plot_data_t6m.rename({'Condition':'Marker'}, axis=1)

fig, axes = plt.subplots(2,3, figsize=(20,10))
axes = axes.ravel()
i = 0
for marker_list, list_name in zip(
    [pan,zone1,perip,zone3,zone2],
    ['Pan zones','Zone 1 centric', 'Periportal', 'Zone 3 centric','Zone 2 and sparce']):
    marker_list = [x.upper() for x in marker_list]
    _plot_data = plot_data_t6m[plot_data_t6m.Marker.isin(marker_list)]
    # g = sns.FacetGrid(_plot_data,col='Condition',col_wrap=4, sharey=False)
    # g = g.map(sns.lineplot, 'zone','percent of cellular tomato area in zone no exp')
    g = sns.lineplot(
        x='zone', y='percent of cellular tomato area in zone no exp', hue='Marker',
        data = _plot_data, ax=axes[i])
    g.axes.set_xlabel('Goz zones')
    g.axes.set_ylabel('')
    g.axes.set_xticklabels([])
    g.axes.set_title(list_name)
    g.axes.legend(bbox_to_anchor = (1,.5), loc='center left', fontsize=20)
    i+=1
fig.delaxes(axes[i])
plt.tight_layout()
plt.savefig('/home2/s190548/work_zhu/images/New Tif/Figure3b.pdf')
plt.close()


# Fig 4a
plot_data = pd.DataFrame()
for _df,_t in zip([plot_data_t0,plot_data_t6m],['Normal','6 Month']):
    _df['T'] = _t
    plot_data = plot_data.append(_df)

g = sns.FacetGrid(
    plot_data,col='Marker',hue='T',col_wrap=5,sharey=False, aspect=1.3,legend_out = False,
    # col_order = [x.upper() for x in (pan + zone1 + perip + zone3 + zone2)],
    )
g = g.map(sns.lineplot, 'zone','percent of cellular tomato area in zone no exp')
g.set_titles(col_template='{col_name}')
g.set_xticklabels([])
g.set_xlabels('')
g.set_ylabels('')
plt.legend(bbox_to_anchor = (1.2,0.5), loc='center left')
g.savefig('/home2/s190548/work_zhu/images/New Tif/T0 vs T6m profiles.pdf')

################################################################################
# Hamp2 T1w vs T28w figure

marker = 'HAMP2'
hamp2 = plot_data[plot_data.Marker==marker].sort_values('zone')
means = hamp2.groupby(['zone','T']).mean()[
    'percent of cellular tomato area in zone no exp'].reset_index()
std = hamp2.groupby(['zone','T']).std()[
    'percent of cellular tomato area in zone no exp'].reset_index()
count = hamp2.groupby(['zone','T'])[
    'percent of cellular tomato area in zone no exp'].count().reset_index()
pararms = []
for x in means['T'].unique():
    for df in [means, std, count]:
        pararms.append(df[df['T']==x].iloc[:,2].values)

means.set_index('zone', inplace=True)
pvals = ttest_ind_from_stats(*pararms)[1]
diff = means.iloc[::2,1] - means.iloc[::-2,1][::-1]
diff.plot(kind = 'bar')
plt.xlabel('GOZ Zone')
plt.ylabel('Difference between % of Tomato area in zones')
plt.xticks([1,8,16,24],['1', '8', '16', '24'])
plt.savefig('HMAP2 diff T1w vs T28w.pdf')
################################################################################

# Figure 4b and supplemental S1

cond_folders = [
    '/home2/s190548/work_zhu/images/New Tif/Normal  30%',
    '/home2/s190548/work_zhu/images/New Tif/6 months 30%']
markers = ['Hamp2','Mup3','Tert']
pooled_data = pd.DataFrame()
for cond,cond_value in zip(cond_folders,['Normal','6 Months']):
    cond = [cond]
    clonal_size_df = get_pooled_zonal_data(cond, markers,'spot clonal sizes.csv')
    clonal_size_df['marker'] = clonal_size_df.Condition.apply(lambda x: x.split(' ')[-1].upper())
    clonal_size_df.Condition = cond_value
    pooled_data = pooled_data.append(clonal_size_df)

# plot zone abs clone counts
sns.set(style='white',font_scale=2, context='paper')
for marker in pooled_data.marker.unique():
    if marker in ['MUP3','TERT']:
        bins = [0,1,2,99]
    else:
        bins = [0,1,4,7,99]
    _marker_pooled_data = pooled_data[pooled_data.marker == marker]
    _binned_data = binning_clonal_sizes(_marker_pooled_data, bins=bins)
    _marker_data_abs_mean = pd.DataFrame()
    zones = np.arange(1,25)
    clonal_sizes = _binned_data.clonal_sizes.astype(str).unique()
    multiidx = product(zones,clonal_sizes)
    multiidx = pd.MultiIndex.from_tuples(multiidx,names=['zone','clonal_sizes'])
    
    for df_g in _binned_data.groupby(['marker','Condition']):
        _df = df_g[1]
        cond, marker = _df.iloc[0,[2,4]]
        _df.clonal_sizes = _df.clonal_sizes.astype(str)
        _absolute_count = _df.groupby(
            ['zone','clonal_sizes','batch']
            ).sum().fillna(0)
        _absolute_count = _absolute_count.groupby(
            ['zone','clonal_sizes']).sum() / _df.batch.nunique()
        _absolute_count = _absolute_count.reindex(multiidx).fillna(0).reset_index()
        # Temp code start 
        # _absolute_count.zone = (_absolute_count.zone + 2) //3
        # _absolute_count = _absolute_count.groupby(['zone','clonal_sizes']).mean()
        # Temp code end
        _absolute_count['Condition'] = cond
        _absolute_count['marker'] = marker
        _marker_data_abs_mean = _marker_data_abs_mean.append(_absolute_count)
        # plotting
        plot_data = _marker_data_abs_mean.reset_index().copy()
        fig, axes = plt.subplots(
            1, 2, sharey=True,figsize=(16,5), gridspec_kw={"wspace":0.02})
        axes = axes.ravel()
        fig.set_tight_layout(True)
        i=0
        plot_legend = False
        for cond in plot_data.Condition.unique()[::-1]:
            _plot_data = plot_data[plot_data.Condition==cond]
            plot_spot_clonal_sizes(
                _plot_data,title='-'.join([marker, cond]),
                ylab= 'Average number of clones in zone ', 
                ax = axes[i], legend=plot_legend)
            i += 1
            plot_legend = True
        fig.savefig(marker + ' spot clonal size.pdf',bbox_inches='tight')
        plt.close(fig)

################################################################################
'''
for marker in pooled_data.marker.unique():
    bins = [0,1,99]
    _marker_pooled_data = pooled_data[pooled_data.marker == marker]
    _binned_data = binning_clonal_sizes(_marker_pooled_data, bins=bins)
    _marker_data_abs_mean = pd.DataFrame()
    zones = np.arange(1,25)
    clonal_sizes = _binned_data.clonal_sizes.astype(str).unique()
    multiidx = product(zones,clonal_sizes)
    multiidx = pd.MultiIndex.from_tuples(multiidx,names=['zone','clonal_sizes'])
    _marker_data_ratio = pd.DataFrame()
    for df_g in _binned_data.groupby(['marker','Condition']):
        _df = df_g[1]
        cond, marker = _df.iloc[0,[2,4]]
        _df.clonal_sizes = _df.clonal_sizes.astype(str)
        _ratio = _df.groupby(['zone','batch','clonal_sizes']).sum().fillna(0)
        _ratio = _ratio.groupby(
            ['zone','batch']).filler.apply(lambda x: x/x.sum()).dropna()
        _ratio = _ratio.reset_index()
        _ratio['Condition'] = cond
        _ratio['marker'] = marker
        _marker_data_ratio = _marker_data_ratio.append(_ratio)
    # plotting
    _plot_data = _marker_data_ratio[_marker_data_ratio.clonal_sizes=='>=2']
    _plot_data.sort_values('Condition', ascending=False, inplace=True)
    _ = plt.figure(figsize=(16,6))
    sns.boxplot(
        data=_plot_data, x='zone',y='filler',hue='Condition')
    plt.ylabel('Ratio of expanded clones')
    plt.xlabel('GOZ Zones')
    plt.title(marker)
    plt.savefig(marker + ' spot expanded ratio.pdf',bbox_inches='tight')
    plt.close(fig)
'''
'''
binned_data = binning_clonal_sizes(pooled_data, [0,1,2,99])
plot_data_ratio = pd.DataFrame()
for df_g in binned_data.groupby(['marker','Condition']):
    _df = df_g[1]
    marker, cond = df_g[0]
    _ratio = _df.groupby(['zone','batch','clonal_sizes']).sum().fillna(0)
    _ratio = _ratio.groupby(['zone','batch']).filler.apply(lambda x: x/x.sum()).dropna()
    _ratio = _ratio.reset_index()
    _ratio['Condition'] = cond
    _ratio['marker'] = marker
    plot_data_ratio = plot_data_ratio.append(_ratio)

# getting statistics
cond_comparison = pd.DataFrame()
for marker in plot_data_ratio.marker.unique():
    _plot_data = plot_data_ratio[plot_data_ratio.marker == marker].copy()
    for gp in _plot_data.groupby(['zone','clonal_sizes']):
        _df = gp[1]
        zone, clonal_size = gp[0]
        if _df.Condition.nunique() == 1:
            continue
        elif len(_df) < 6:
            continue
        normals = _df.loc[_df.Condition == 'Normal','filler'].values
        m6s = _df.loc[_df.Condition == '6 Months','filler'].values
        pval = ttest_ind(normals,m6s)[1]
        cond_comparison = cond_comparison.append({
            "zone": zone,
            "marker": marker,
            "clonal_size": clonal_size,
            "pvalue": pval
        }, ignore_index=True)
cond_comparison.to_csv('Condition wise ttest per zone per clonal size.csv')
'''

# Figure 5c

os.chdir("/home2/s190548/work_zhu/images/New Tif")
_=sns.plotting_context('paper', font_scale = 2)
plt.rcParams['font.serif'] = 'Arial'
cond_folders = [
    '/home2/s190548/work_zhu/images/New Tif/big tif']
markers = ['AAV', 'Tert', 'MUP3']
plot_data = get_pooled_zonal_data(cond_folders, markers)
plot_data = plot_data[plot_data.batch == 'batch1']
plot_data.replace('big tif 2019_11_26__1080', 'MUP3', inplace=True)
plot_data.replace('big tif ', '', inplace=True, regex=True)
plot_data.zone = plot_data.zone.str.zfill(2)
plot_data = plot_data.sort_values("zone")
for _df in plot_data.groupby('Condition'):
    marker = _df[0]
    _df = _df[1]
    plot_pooled_zonal_data(
        _df, 
        y_col='percent of cellular tomato area in zone no exp', 
        plot_diff_bar=False,
        )
    plt.xlabel('')
    plt.xticks([])
    plt.ylim((0,2))
    plt.legend('')
    figname = marker + ' big tif figure.pdf'
    plt.savefig(figname,bbox_inches='tight')
    plt.close()


'''# Figure 6c

cond_folders = [
    '/home2/s190548/work_zhu/images/New Tif/Normal  30%',
    '/home2/s190548/work_zhu/images/New Tif/6 months 30%']
markers = ['Hamp2','Mup3','Tert']
plot_data = pd.DataFrame()
for cond,cond_value in zip(cond_folders,['Normal','6 Months']):
    cond = [cond]
    clonal_size_df = get_pooled_zonal_data(cond, markers,'spot clonal sizes.csv')
    clonal_size_df['marker'] = clonal_size_df.Condition.apply(lambda x: x.split(' ')[-1].upper())
    clonal_size_df.Condition = cond_value
    plot_data = plot_data.append(clonal_size_df)

invalid_obs = []
plot_data.index = ['spot_' + str(x+1) for x in range(plot_data.shape[0])]
for _group in plot_data.groupby(['Condition','marker','zone']):
    _df = _group[1]
    iqr = _df.spot_size_d.quantile(0.75) - _df.spot_size_d.quantile(0.25)
    upper = _df.spot_size_d.quantile(0.75) + 1.5 * iqr
    lower = _df.spot_size_d.quantile(0.25) - 1.5 * iqr
    _invalid = _df[
        (_df.spot_size_d > upper) | (_df.spot_size_d <lower)
        ].index.tolist()
    if _invalid != []:
        invalid_obs += _invalid
plot_data = plot_data.drop(invalid_obs)

_ = plt.figure(figsize=(12,6))
g = sns.FacetGrid(
    plot_data,col='marker',hue='Condition', sharey=True, aspect=1.3,
    legend_out = False,
    )
g = g.map(sns.lineplot, 'zone','spot_size_d')
g.set_titles(col_template='{col_name}')
g.set_xticklabels([])
g.set_xlabels('')
g.set_ylabels('')
plt.legend(bbox_to_anchor = (1.2,0.5), loc='center left')
plt.tight_layout()
plt.savefig('Spot size in zones.pdf')'''

'''
hamp2 = plot_data[
    (plot_data.marker == 'HAMP2') & (plot_data.Condition=='Normal')
    ].groupby(['zone','batch']).mean().reset_index()
_ = plt.figure(figsize=(16,8))
sns.swarmplot(data = hamp2, y='spot_size_d',x='zone')

mup3 = plot_data[
    (plot_data.marker == 'MUP3') & (plot_data.Condition=='Normal')
    ]
_ = plt.figure(figsize=(16,8))
sns.swarmplot(data = mup3, y='spot_size_d',x='zone')

tert = plot_data[
    (plot_data.marker == 'TERT') & (plot_data.Condition=='Normal')
    ]
tert = tert.groupby(['zone','batch']).mean().reset_index()
_ = plt.figure(figsize=(16,8))
sns.swarmplot(data = tert, y='spot_size_d',x='zone')

tert = plot_data[
    (plot_data.marker == 'TERT') & (plot_data.Condition=='6 Months')
    ]
tert = tert.groupby(['zone','batch']).mean().reset_index()
_ = plt.figure(figsize=(16,8))
sns.swarmplot(data = tert, y='spot_size_d',x='zone')


hamp2 = plot_data[
    (plot_data.marker == 'HAMP2')
    ]
_ = plt.figure(figsize=(16,8))
sns.swarmplot(data = hamp2, y='spot_size_d',x='Condition')

'''