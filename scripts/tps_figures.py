#%%
import pandas as pd
import skimage as ski
from skimage import io
import os
from tps.segmentation import *
from tps.plotting import *
from tps.find_zones import *
from scipy.stats import ttest_ind, ttest_ind_from_stats
from itertools import product
import seaborn as sns

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
    g.set_xlabel('TPS Layer')
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

#%%
input_folder = "/endosome/work/InternalMedicine/s190548/TPS/tps_results"
output_folder = "/endosome/work/InternalMedicine/s190548/TPS/tps_results/figures"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder)
# manifest = pd.read_csv(
#     os.path.join(input_folder, 'manifest.txt'), sep ='\t', index_col=0)
# _ = sns.plotting_context('paper', font_scale = 1.5)
sns.mpl.rcParams['font.family'] = ['Arial']
sns.mpl.rcParams['font.serif'] = ['Arial']
sns.mpl.rcParams['font.monospace'] = ['Arial']
sns.mpl.rcParams['font.fantasy'] = ['Arial']
sns.mpl.rcParams['font.weight'] = 'bold'
sns.set_theme('paper', font_scale = 2, font='Arial', style='white')
#%%
# bad_folders = manifest[manifest.skip==1].index
# for bf in bad_folders:
#     bf = '/'.join(bf.split('/')[:-1])
#     os.system('mv {} /endosome/archive/shared/zhu_wang/TPS_paper/TIF-20211214/temp'.format(bf))

# bad_folders = manifest[manifest.skip==0.5].index
# for bf in bad_folders:
#     bf = '/'.join(bf.split('/')[:-1])
#     os.system('mv {} /endosome/archive/shared/zhu_wang/TPS_paper/TIF-20211214/temp'.format(bf))
#%%
# Figure 2e

files = [
    '/endosome/archive/shared/zhu_wang/TPS_paper/TIF-20211214/2w_Control/Gs-0287-m-L3-1/zone int.csv',
    '/endosome/archive/shared/zhu_wang/TPS_paper/TIF-20211214/2w_Control/Cyp1a2-0301-f345-L1/zone int.csv',
    '/endosome/archive/shared/zhu_wang/TPS_paper/TIF-20211214/2w_Control/Gls2-0244-F456-L2/zone int.csv',
]
plot_data = pd.DataFrame()
for marker, fn in zip(['GS','CYP1A2','GSL2'],files):
    _df = pd.read_csv(fn, index_col=0)
    _df['Marker'] = marker
    _df.zone = [x.replace("Z", "") for x in _df.zone.values]
    _df.zone = _df.zone.str.zfill(2)
    _df = _df.sort_values('zone')
    plot_data = plot_data.append(_df)

g = sns.FacetGrid(
    plot_data,col='Marker',sharey=True, aspect=1.3,
    )
g = g.map(sns.lineplot, 'zone','percent of cellular tomato area in zone no exp')
g.set_titles(col_template='')
g.set_xticklabels([])
g.set_xlabels('TPS Layers')
g.set_ylabels('% Tomato area in layer')
plt.legend(bbox_to_anchor = (1.2,0.5), loc='center left')
plt.savefig(output_folder + '/Figure 2e.pdf', bbox_inches='tight')

#%%

############################################################################
# Figure 3b
cond_folder = input_folder + '/2w'
markers = []
markers += [x.split("-")[0] for x in os.listdir(cond_folder) if "tif" in x]
markers = list(set(markers))
plot_data_t0 = get_pooled_zonal_data([cond_folder], markers)
plot_data_t0.zone = [x.replace("Z", "") for x in plot_data_t0.zone.values]
plot_data_t0.zone = plot_data_t0.zone.str.zfill(2)
plot_data_t0 = plot_data_t0.sort_values("zone")
plot_data_t0.Condition = plot_data_t0.Condition.apply(lambda x: x.split(' ')[-1].upper())
plot_data_t0 = plot_data_t0.sort_values('Condition')
plot_data_t0 = plot_data_t0.rename({'Condition':'Marker'}, axis=1)
zone_order = sorted(plot_data_t0.zone.unique())
plot_data_t0 = plot_data_t0[~plot_data_t0.zone.isin(['01','24'])]
plot_data_t0.Marker = plot_data_t0.Marker.str.replace('2W_CONTROL_','')
pan = ['Apoc4','Pklr']
zone1 = ['Arg1.1', 'Arg1.2','Gls2']
perip = ['Sox9','Krt19']
zone3 = ['Gs','Cyp1a2','Oat', 'Axin2']
zone2 = ['Hamp2','Mup3','Tert']

fig, axes = plt.subplots(2,3, figsize=(30,15))
sns.set_theme('paper', font_scale = 3, font='Arial', style='white')
axes = axes.ravel()
i = 0
for marker_list, list_name in zip(
    [pan,zone1,perip,zone3,zone2],
    ['Pan zones','Zone 1 centric', 'Periportal', 'Zone 3 centric','Zone 2 and sparce']):
    marker_list = [x.upper() for x in marker_list]
    _plot_data = plot_data_t0[plot_data_t0.Marker.isin(marker_list)]
    _plot_data = _plot_data.sort_values('zone')
    # g = sns.FacetGrid(_plot_data,col='Condition',col_wrap=4, sharey=False)
    # g = g.map(sns.lineplot, 'zone','percent of cellular tomato area in zone no exp')
    g = sns.lineplot(
        x='zone', y='percent of cellular tomato area in zone no exp', hue='Marker',
        data = _plot_data, ax=axes[i])
    g.axes.set_xlabel('TPS Layers', fontsize = 'large')
    g.axes.set_ylabel('')
    g.axes.set_xticklabels([])
    g.axes.set_title(list_name, fontsize = 'large')
    g.axes.legend(bbox_to_anchor = (1,.5), loc='center left', fontsize=20)
    i+=1
fig.delaxes(axes[i])
plt.tight_layout()
plt.savefig('Figure 3b.pdf')
plt.close()

#%%
#############################################################################
# Figure 3c
cond_folder = input_folder + '/48w'
markers = []
markers += [x.split("-")[0] for x in os.listdir(cond_folder) if "tif" in x]
markers = list(set(markers))
plot_data_t6m = get_pooled_zonal_data([cond_folder], markers)
plot_data_t6m.zone = [x.replace("Z", "") for x in plot_data_t6m.zone.values]
plot_data_t6m.zone = plot_data_t6m.zone.str.zfill(2)
plot_data_t6m = plot_data_t6m.sort_values("zone")
plot_data_t6m.Condition = plot_data_t6m.Condition.apply(lambda x: x.split(' ')[-1].upper())
plot_data_t6m = plot_data_t6m.sort_values('Condition')
plot_data_t6m = plot_data_t6m.rename({'Condition':'Marker'}, axis=1)
zone_order = sorted(plot_data_t6m.zone.unique())
plot_data_t6m = plot_data_t6m[~plot_data_t6m.zone.isin(['01','24'])]
plot_data_t6m.Marker = plot_data_t6m.Marker.str.replace('6_MONTHS_','')
fig, axes = plt.subplots(2,3, figsize=(30,15))
sns.set_theme('paper', font_scale = 3, font='Arial', style='white')
axes = axes.ravel()
i = 0
for marker_list, list_name in zip(
    [pan,zone1,perip,zone3,zone2],
    ['Pan zones','Zone 1 centric', 'Periportal', 'Zone 3 centric','Zone 2 and sparce']):
    marker_list = [x.upper() for x in marker_list]
    _plot_data = plot_data_t6m[plot_data_t6m.Marker.isin(marker_list)]
    _plot_data = _plot_data.sort_values('zone')
    # g = sns.FacetGrid(_plot_data,col='Condition',col_wrap=4, sharey=False)
    # g = g.map(sns.lineplot, 'zone','percent of cellular tomato area in zone no exp')
    g = sns.lineplot(
        x='zone', y='percent of cellular tomato area in zone no exp', hue='Marker',
        data = _plot_data, ax=axes[i])
    g.axes.set_xlabel('TPS Layers', fontsize = 'large')
    g.axes.set_ylabel('')
    g.axes.set_xticklabels([])
    g.axes.set_title(list_name,fontsize = 'large')
    g.axes.legend(bbox_to_anchor = (1,.5), loc='center left', fontsize=20)
    i+=1
fig.delaxes(axes[i])
plt.tight_layout()
plt.savefig('Figure 3c.pdf')
plt.close()

#%%
# Fig 5a
plot_data = pd.DataFrame()
for _df,_t in zip([plot_data_t0,plot_data_t6m],['Normal','6 Month']):
    _df['T'] = _t
    plot_data = plot_data.append(_df)
plot_data = plot_data[plot_data.Marker!='TERT']
plot_data = plot_data.sort_values('zone')
plot_markers = [x.upper() for x in (pan + zone1 + perip + zone3 + zone2)]
plot_markers.remove('TERT')
_ = plt.figure(figsize=(30,15))
g = sns.FacetGrid(
    plot_data,col='Marker',hue='T',col_wrap=5,sharey=False, aspect=1,
    legend_out = False, height=5,
    col_order = plot_markers,
    )
g = g.map(sns.lineplot, 'zone','percent of cellular tomato area in zone no exp')
g.set_titles(col_template='{col_name}',fontsize = 'large')
g.set_xticklabels([])
g.set_xlabels('')
g.set_ylabels('')
plt.legend(bbox_to_anchor = (1.2,0.5), loc='center left')
g.savefig('Figure 5a.pdf')
plt.close()
#%%
# # Figure 5a alternative

# g = sns.FacetGrid(
#     plot_data,col='Marker',hue='T',col_wrap=5,sharey=False, aspect=1,
#     height=4, legend_out = False, gridspec_kws = {'wspace':0.01,'hspace':0.01}
#     # col_order = [x.upper() for x in (pan + zone1 + perip + zone3 + zone2)],
#     )
# g = g.map(sns.lineplot, 'zone', 'percent of positive in zone')
# g.set_titles(col_template='{col_name}')
# g.set_xticklabels([])
# g.set_xlabels('')
# g.set_ylabels('')
# g.fig.tight_layout()
# plt.legend(bbox_to_anchor = (1.2,0.5), loc='center left')
# g.savefig('Figure 5a alternative.pdf')

#%%
# zonal summary table
multiidx = product(
    plot_data.Marker.unique(),
    plot_data['T'].unique(),)
multiidx = pd.MultiIndex.from_tuples(multiidx,names=['Marker','Condition'])
res = pd.DataFrame(index=multiidx)
n_zones = 24
for g in plot_data.groupby(['Marker','T']):
    marker, condition = g[0]
    _df = g[1]
    _df = _df.groupby('zone').mean()
    _int = _df['percent of positive in zone']
    z_max = (1-int(_int.idxmax())/n_zones) * 2 + 1
    z_50 = (_int <= _int.max()/2).sum()/n_zones
    res.loc[(marker, condition), 'Zone'] = z_max
    res.loc[(marker, condition), 'IZ50'] = z_50
res.to_csv('Marker Zmax IZ50 summary.csv')
#%%

# # Figure 5b
# # Hamp2 T1w vs T28w figure

# marker = 'HAMP2'
# hamp2 = plot_data[plot_data.Marker==marker].sort_values('zone')
# means = hamp2.groupby(['zone','T']).mean()[
#     'percent of cellular tomato area in zone no exp'].reset_index()
# std = hamp2.groupby(['zone','T']).std()[
#     'percent of cellular tomato area in zone no exp'].reset_index()
# count = hamp2.groupby(['zone','T'])[
#     'percent of cellular tomato area in zone no exp'].count().reset_index()
# pararms = []
# for x in means['T'].unique():
#     for df in [means, std, count]:
#         pararms.append(df[df['T']==x].iloc[:,2].values)

# means.set_index('zone', inplace=True)
# pvals = ttest_ind_from_stats(*pararms)[1]
# diff = means.iloc[::2,1] - means.iloc[::-2,1][::-1]
# diff.plot(kind = 'bar')
# plt.xlabel('TPS Layers')
# plt.ylabel('Difference between % of Tomato area in zones')
# plt.xticks([1,8,16,24],['1', '8', '16', '24'])
# plt.savefig('Figure 5b')

# means = hamp2.groupby('T').mean()[
#     'percent of cellular tomato area in zone no exp'].reset_index()
# std = hamp2.groupby('T').std()[
#     'percent of cellular tomato area in zone no exp'].reset_index()
# count = hamp2.groupby('T')[
#     'percent of cellular tomato area in zone no exp'].count().reset_index()
# pararms = []
# for x in means['T'].unique():
#     for df in [means, std, count]:
#         pararms.append(df[df['T']==x].iloc[:,1].values)
# pvals = ttest_ind_from_stats(*pararms)[1]

#%%

################################################################################

# Figure 5b and supplemental S1

cond_folders = ['2w','48w']
cond_folders = [os.path.join(input_folder, x) for x in cond_folders]

markers = ['Hamp2','Mup3','Tert']
pooled_data = pd.DataFrame()
for cond,cond_value in zip(cond_folders,['Normal','6 Months']):
    cond = [cond]
    clonal_size_df = get_pooled_zonal_data(cond, markers,'spot clonal sizes.csv')
    clonal_size_df['marker'] = clonal_size_df.Condition.apply(lambda x: x.split(' ')[-1].upper())
    clonal_size_df.Condition = cond_value
    pooled_data = pooled_data.append(clonal_size_df)

# plot zone abs clone counts
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
    
    for df_g in _binned_data.groupby('Condition'):
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
                ylab= 'Average number of clones in layer ', 
                ax = axes[i], legend=plot_legend)
            i += 1
            plot_legend = True
        fig.savefig(marker + ' spot clonal size.pdf',bbox_inches='tight')
        plt.close(fig)

os.rename('HAMP2 spot clonal size.pdf','Figure 5b.pdf')
os.rename('MUP3 spot clonal size.pdf','Figure S3.pdf')

#%%

################################################################################

# Figure 5c expanded clones

for marker in ['HAMP2','MUP3']:
    if marker == 'HAMP2':
        bins = [0,5,99]
    else:
        bins = [0,2,99]
    _marker_pooled_data = pooled_data[pooled_data.marker == marker]
    _binned_data = binning_clonal_sizes(_marker_pooled_data, bins=bins)
    _marker_data_abs_mean = pd.DataFrame()
    zones = np.arange(1,25)
    batches = _binned_data.batch.astype(str).unique()
    clonal_sizes = _binned_data.clonal_sizes.astype(str).unique()
    multiidx = product(zones, batches, clonal_sizes)
    multiidx = pd.MultiIndex.from_tuples(
        multiidx,names=['zone','batch','clonal_sizes'])
    _marker_data_ratio = pd.DataFrame()
    for df_g in _binned_data.groupby('Condition'):
        _df = df_g[1]
        cond, marker = _df.iloc[0,[2,4]]
        _df.clonal_sizes = _df.clonal_sizes.astype(str)
        _ratio = _df.groupby(['zone','batch','clonal_sizes']).sum().fillna(0)
        _ratio = _ratio.reindex(multiidx).fillna(0)
        _ratio = _ratio.groupby(
            ['zone','batch']).filler.apply(lambda x: x/x.sum()).dropna()
        _ratio = _ratio.reset_index()
        _ratio['Condition'] = cond
        _ratio['marker'] = marker
        _marker_data_ratio = _marker_data_ratio.append(_ratio)
    # plotting
    expanded = _marker_data_ratio.clonal_sizes.unique()[1]
    _plot_data = _marker_data_ratio[_marker_data_ratio.clonal_sizes==expanded]
    _plot_data.sort_values('Condition', ascending=False, inplace=True)
    _ = plt.figure(figsize=(16,6))
    colors = sns.diverging_palette(30,270,n=2, s=100)
    sns.swarmplot(
        data=_plot_data, x='zone',y='filler',hue='Condition', alpha=0.5,
        s = 10, palette = colors)
    plt.ylabel('Ratio of expanded clones to all clones')
    plt.xlabel('TPS Layers')
    plt.title(marker)
    plt.savefig(marker + ' spot expanded ratio.pdf',bbox_inches='tight')
    plt.close(fig)

os.rename('HAMP2 spot expanded ratio.pdf','Figure 5c.pdf')

#%%
# AAV GOZ pattern
cond_folders = ['/home2/s190548/work_zhu/images/New Tif/big tif']
markers = ['AAV']
plot_data_aav = get_pooled_zonal_data(cond_folders, markers)
plot_data_aav.zone = [x.replace("Z", "") for x in plot_data_aav.zone.values]
plot_data_aav.zone = plot_data_aav.zone.str.zfill(2)
plot_data_aav = plot_data_aav.sort_values("zone")
plot_data_aav.Condition = plot_data_aav.Condition.apply(lambda x: x.split(' ')[-1].upper())
plot_data_aav = plot_data_aav.sort_values('Condition')
plot_data_aav = plot_data_aav.rename({'Condition':'Marker'}, axis=1)

g = sns.lineplot(
    x='zone', y='percent of cellular tomato area in zone no exp',
    data = plot_data_aav)
g.axes.set_xlabel('TPS Layers')
g.axes.set_ylabel('% Tomato area in layer')
g.axes.set_xticklabels([])
plt.savefig('AAV.pdf',bbox_inches='tight')
# plt.tight_layout()
# plt.savefig('/home2/s190548/work_zhu/images/New Tif/Figure 3b.pdf')
# plt.close()

#%%
cond_folder = [input_folder + '/DDC', input_folder + '/DDC_control']
markers = ['Apoc4', 'Gls2', 'GS']
ddc = get_pooled_zonal_data(cond_folder, markers)
ddc.zone = [x.replace("Z", "") for x in ddc.zone.values]
ddc.zone = ddc.zone.str.zfill(2)
ddc = ddc.sort_values('Condition')
ddc = ddc.rename({'Condition':'Marker'}, axis=1)
zone_order = sorted(ddc.zone.unique())
ddc = ddc[~ddc.zone.isin(['01','24'])]
ddc['T'] = [x.split(' ')[0] for x in ddc.Marker]
ddc.Marker = ddc.Marker.str.replace('DDC ','')
ddc.Marker = ddc.Marker.str.replace('DDC_control ','')
ddc = ddc.sort_values("zone")
ddc['T'].replace('DDC_control', 'Control', inplace = True)

g = sns.FacetGrid(
    ddc,col='Marker',hue='T',col_wrap=5,sharey=False, aspect=1,
    legend_out = False, height=5,
    col_order = markers,
    )
g = g.map(sns.lineplot, 'zone','percent of cellular tomato area in zone no exp', lw=2)
g.set_titles(col_template='{col_name}',fontsize = 'large')
g.set_xticklabels([])
g.set_xlabels('')
g.set_ylabels('')
plt.legend(bbox_to_anchor = (1.2,0.5), loc='center left')
g.savefig('Figure 6d.pdf')
plt.close()

ddc = ddc[ddc['batch'] == 'batch2']
ddc['T1'] = ddc.Marker + '_' + ddc['T']
g = sns.FacetGrid(
    ddc, row='T1', sharey=True, aspect=16/9,
    legend_out = False, height=4,
    despine=False, gridspec_kws = {'hspace':0.15}
    )
g.map(sns.lineplot, 'zone','percent of cellular tomato area in zone no exp', lw=2)
g.set_titles('')
g.set_xticklabels([])
g.set_xlabels('TPS Layers')
g.set_ylabels('')
g.savefig('Figure 6c.pdf')
plt.close()
# %%
#%%
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