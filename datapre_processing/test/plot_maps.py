__author__ = 'Marco De Nadai'
__license__ = "MIT"

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import datetime
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap

# Set the plot styles
parse = lambda x: datetime.datetime.fromtimestamp(float(x) / 1000)

fig_width_pt = 345  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27  # Convert pt to inches
golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]

sns.set_style("ticks")
sns.set_context("paper")

# Import the dataset Telecommunications
datadir = r"D:\Myproject\Python\Datasets\MobileFlowData\SourceData\Milan"
file_name = 'sms-call-internet-mi-2013-11-'
sliceSum = pd.DataFrame({})
for index in range(4, 11):
    sliceSum2 = pd.read_csv(os.path.join(datadir, file_name) + str(index).zfill(2) + '.txt', sep='\t',
                            encoding="utf-8-sig",
                            names=['CellID', 'datetime', 'countrycode', 'smsin', 'smsout', 'callin', 'callout',
                                   'internet'], parse_dates=['datetime'],
                            date_parser=parse)  # , parse_dates=['datetime'])
    sliceSum2 = sliceSum2.set_index('datetime')
    sliceSum2['hour'] = sliceSum2.index.hour
    sliceSum2['weekday'] = sliceSum2.index.weekday
    sliceSum2 = sliceSum2.groupby(['hour', 'weekday', 'CellID'], as_index=False).sum()

    sliceSum = sliceSum.append(sliceSum2)

sliceSum['idx'] = sliceSum['hour'] + (sliceSum['weekday'] * 24)
sliceSum.head()

# Import the dataset - precipitations
# precipitation_df = pd.read_csv('nature/precipitation-trentino.csv', sep=',', names=['datetime', 'CellID', 'intensity'],
#                                encoding="utf-8-sig", parse_dates=['datetime'], date_parser=parse)

# precipitation_df = precipitation_df.set_index(['datetime'], drop=False)
# precipitation_df = precipitation_df.groupby(['CellID'], as_index=False).mean()
# precipitation_df.head()

# Import the dataset - news
# news_df = pd.read_csv('nature/news.csv', sep=',', encoding="utf-8-sig", parse_dates=['date'])
# news_df.head()

# Import the grid, obtained from the geojson. The CSV has X, Y coordinates and CellID
# csv_df = pd.read_csv('nature/grid_trentino.csv')

# Merge csv_df with Telecommunications, grouped by cellID
# merge = pd.merge(sliceSum.groupby(['CellID'], as_index=False).sum(), csv_df, on='CellID')

# Extract the points for hexbin
# points_x = np.array([x['X'] for i, x in merge.iterrows()])
# points_y = np.array([x['Y'] for i, x in merge.iterrows()])
# c = np.array([x['internet'] for i, x in merge.iterrows()])

# Trentino's boundingbox
a = (45.6730682227551, 10.4521594968354)
b = (46.5327699992773, 11.9627133503828)

# Trentino shapefile
# http://dati.trentino.it/dataset/limite-comprensoriale-027140/resource/ff1f1687-3f8f-427e-84d9-cf40c8b9b98a
m = Basemap(lat_0=(a[0] + b[0]) / 2, lon_0=(a[1] + b[1]) / 2, epsg=4326, llcrnrlon=a[1], llcrnrlat=a[0], urcrnrlon=b[1],
            urcrnrlat=b[0], )
m.readshapefile('nature/amm', 'Trentino_shapefile', color='0.35')

cmap = LinearSegmentedColormap.from_list("skil", sns.color_palette("RdBu_r", 7)[1:])
plt.register_cmap(cmap=cmap)
# m.hexbin(points_x, points_y, cmap="skil", gridsize=50, C=c, bins='log', mincnt=1)
sns.despine(left=True, bottom=True)
plt.savefig('map.pdf', format='pdf', dpi=330, bbox_inches='tight')

# Import the dataset - social pulse, obtained from the geojson
# social_df = pd.read_csv('nature/result2.csv', sep=',', encoding="utf-8-sig", parse_dates=['created'])
# points_x = np.array([x['geomPoint.geom/coordinates/0'] for i, x in social_df.iterrows()])
# points_y = np.array([x['geomPoint.geom/coordinates/1'] for i, x in social_df.iterrows()])

m = Basemap(lat_0=(a[0] + b[0]) / 2, lon_0=(a[1] + b[1]) / 2, epsg=4326, llcrnrlon=a[1], llcrnrlat=a[0], urcrnrlon=b[1],
            urcrnrlat=b[0], )
m.readshapefile('nature/amm', 'Trentino_shapefile', color='0.35')

# m.hexbin(points_x, points_y, cmap="skil", gridsize=50, bins='log', mincnt=1)
sns.despine(left=True, bottom=True)
plt.savefig('map_social.pdf', format='pdf', dpi=330, bbox_inches='tight')

# Energy map
line_df = pd.read_csv('nature/line.csv', sep=',', encoding="utf-8-sig")
line_df['CellID'] = line_df['SQUAREID']
# merge = pd.merge(line_df.groupby(['CellID'], as_index=False).sum(), csv_df, on='CellID')
# points_x = np.array([x['X'] for i, x in merge.iterrows()])
# points_y = np.array([x['Y'] for i, x in merge.iterrows()])
# c = np.array([x['NR_UBICAZIONI'] for i, x in merge.iterrows()])

m = Basemap(lat_0=(a[0] + b[0]) / 2, lon_0=(a[1] + b[1]) / 2, epsg=4326, llcrnrlon=a[1], llcrnrlat=a[0], urcrnrlon=b[1],
            urcrnrlat=b[0], )
m.readshapefile('nature/amm', 'Trentino_shapefile', color='0.35')

# m.hexbin(points_x, points_y, cmap="skil", gridsize=50, bins='log', C=c, mincnt=1)
sns.despine(left=True, bottom=True)
plt.savefig('map_line.pdf', format='pdf', dpi=330, bbox_inches='tight')

# Precipitation map
# merge = pd.merge(precipitation_df, csv_df, on='CellID')
# points_x = np.array([x['X'] for i, x in merge.iterrows()])
# points_y = np.array([x['Y'] for i, x in merge.iterrows()])
# c = np.array([x['intensity'] for i, x in merge.iterrows()])

m = Basemap(lat_0=(a[0] + b[0]) / 2, lon_0=(a[1] + b[1]) / 2, epsg=4326, llcrnrlon=a[1], llcrnrlat=a[0], urcrnrlon=b[1],
            urcrnrlat=b[0], )
m.readshapefile('nature/amm', 'Trentino_shapefile', color='0.35')

# m.hexbin(points_x, points_y, cmap="skil", gridsize=50, bins='log', C=c, mincnt=1)
sns.despine(left=True, bottom=True)
plt.savefig('map_precipitation.pdf', format='pdf', dpi=330, bbox_inches='tight')

# News map
# points_x = np.array([x['geomPoint.geom/coordinates/0'] for i, x in news_df.iterrows()])
# points_y = np.array([x['geomPoint.geom/coordinates/1'] for i, x in news_df.iterrows()])

m = Basemap(lat_0=(a[0] + b[0]) / 2, lon_0=(a[1] + b[1]) / 2, epsg=4326, llcrnrlon=a[1], llcrnrlat=a[0], urcrnrlon=b[1],
            urcrnrlat=b[0], )
m.readshapefile('nature/amm', 'Trentino_shapefile', color='0.35')

# m.hexbin(points_x, points_y, cmap="skil", gridsize=35, bins='log', mincnt=1)
sns.despine(left=True, bottom=True)
plt.savefig('map_news.pdf', format='pdf', dpi=330, bbox_inches='tight')
