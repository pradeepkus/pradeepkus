#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:40:33 2021

@author: pradeep
"""
print(__doc__)
import numpy as np
# from sklearn import metrics
# from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn import datasets
from sklearn.cluster import KMeans
# #from mpl_toolkits import Basemap
# import netCDF4 as nc
import xarray as xr
# from   datetime import timedelta
# import numpy as geek 
# import cartopy.crs as ccrs
from sklearn import metrics
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from matplotlib.pyplot import *

fn        = '/home/pradeep/Desktop/Dropbox/PAPER2_REVISED4/DATA/KMEAN_INPUT_H_600_15N_22_68_72E_7.6mm.nc'
dset      = xr.open_dataset(fn)
OLR       = dset['KFILTERED_H'][:][:][:]
OLR1      = OLR
time      = dset['time'][:]
lat       = dset['lat'][:]
lon       = dset['lon'][:]
xreshape  = dset['xreshape'][:][:]
scaled_features=StandardScaler().fit_transform(xreshape)

#========================================================================
fig, ax = plt.subplots(2, 2, figsize=(12,8))
for i in [ 2, 3, 4, 5]:
    '''
    Create KMeans instance for different number of clusters
    '''
    km = KMeans(n_clusters=i, init='k-means++', n_init=20, max_iter=20,random_state=0)
    q, mod = divmod(i, 2)
    '''
    Create SilhouetteVisualizer instance with KMeans instance
    Fit the visualizer
    '''
    visualizer = SilhouetteVisualizer(km,colors='yellowbrick',ax=ax[q-1][mod]) #(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(scaled_features)
    visualizer.finalize()      # Finalize and render the figure
   #labels     = visualizer.y_tick_pos_
   #visualizer.draw(labels) 
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
   # plt.set_xlabel("x-label", fontsize=12)
    plt.savefig('Silhouette_Score_Density.pdf',dpi=10)
   # plt.title('Silhouette Score Density')
# from sklearn.cluster import KMeans

# from yellowbrick.cluster import SilhouetteVisualizer
# from yellowbrick.datasets import load_nfl

# # Load a clustering dataset
# X, y = load_nfl()

# # Specify the features to use for clustering
# features = ['Rec', 'Yds', 'TD', 'Fmb', 'Ctch_Rate']
# X = X.query('Tgt >= 20')[features]

# # Instantiate the clustering model and visualizer
# model = KMeans(5, random_state=42)
    
