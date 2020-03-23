import pandas as pd
import skimage as ski
from skimage import io
import os
from goz.segmentation import *
from goz.plotting import *
from goz.find_zones import *

os.chdir("Z:/images/")
_=sns.plotting_context('paper', font_scale = 1.3)
cond_folders = ['Z:/images/New Tif/Normal  30%', 'Z:/images/New Tif/6 months 30%']

markers = []
for cond_folder in cond_folders:
    markers += [x.split("-")[0] for x in os.listdir(cond_folder) if "tif" in x]
markers = list(set(markers))

# # Single colors
# for cond_folder, color in zip(cond_folders, [240, 0]):
#     for marker in markers:
#         plot_data = get_pooled_zonal_data([cond_folder], [marker])
#         if plot_data.shape[0] == 0:
#             continue
#         plot_pooled_zonal_data(
#             plot_data,
#             figname=marker + " " + cond_folder + ".pdf", forced_color=color
#         )

# for marker in markers:
#     plot_data = get_pooled_zonal_data(cond_folders, [marker])
#     if plot_data.shape[0] == 0:
#         continue
#     plot_pooled_zonal_data(
#         plot_data,
#         figname="APOC4 normal 6moth vs DDC.pdf",
#         plot_diff_bar=False
#     )

cond_folders = ['Z:/images/New Tif/6 months 30%']
markers = []
for cond_folder in cond_folders:
    markers += [x.split("-")[0] for x in os.listdir(cond_folder) if "tif" in x]
markers = list(set(markers))
plot_data_t6m = get_pooled_zonal_data(cond_folders, markers)
plot_data_t6m.zone = [x.replace("Z", "") for x in plot_data_t6m.zone.values]
plot_data_t6m.zone = plot_data_t6m.zone.str.zfill(2)
plot_data_t6m = plot_data_t6m.sort_values("zone")
plot_data_t6m.Condition = plot_data_t6m.Condition.apply(lambda x: x.split(' ')[-1].upper())
# plot_data_t6m = plot_data_t6m.sort_values('Condition')
g = sns.FacetGrid(plot_data_t6m,col='Condition',col_wrap=4,sharey=False)
g = g.map(sns.lineplot, 'zone','percent of tomato area in zone')
g.set_titles(col_template='{col_name}')
g.set_xticklabels([])
g.set_xlabels('')
g.savefig('Z:/images/New Tif/6 month profiles.pdf')

cond_folders = ['Z:/images/New Tif/Normal  30%']
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
g = sns.FacetGrid(plot_data_t0,col='Condition',col_wrap=4,sharey=False)
g = g.map(sns.lineplot, 'zone','percent of tomato area in zone')
g.set_titles(col_template='{col_name}')
g.set_xticklabels([])
g.set_xlabels('')
g.savefig('Z:/images/New Tif/T0 profiles.pdf')

plot_data = pd.DataFrame()
for _df,_t in zip([plot_data_t0,plot_data_t6m],['Normal','6 Month']):
    _df['T'] = _t
    plot_data = plot_data.append(_df)

g = sns.FacetGrid(plot_data,col='Condition',hue='T',col_wrap=4,sharey=False)
g = g.map(sns.lineplot, 'zone','percent of tomato area in zone')
g.set_titles(col_template='{col_name}')
g.set_xticklabels([])
g.set_xlabels('')
g.add_legend()
g.savefig('Z:/images/New Tif/T0 vs T6m profiles.pdf')