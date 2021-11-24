#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:39:42 2021

@author: pradeep
"""
#================================================================================
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
from matplotlib.pyplot import figure

#===============================================================================
fn        ='/home/pradeep/Desktop/Dropbox/PAPER2_REVISED4/DATA/KMEAN_INPUT_H_600_15N_22_68_72E_7.6mm.nc'
dset      = xr.open_dataset(fn)
OLR       = dset['KFILTERED_H'][:][:][:]
OLR1      = OLR
time      = dset['time'][:]
lat       = dset['lat'][:]
lon       = dset['lon'][:]
xreshape  = dset['xreshape'][:][:]
#kmeans   = KMeans(n_clusters=4, init='k-means++', n_init=30, max_iter=100)
scaled_features=StandardScaler().fit_transform(xreshape)
# f = plt.figure() 
# f.set_figwidth(4) 
# f.set_figheight(1) 
# #============================================================================
silhouette_coefficients = []
for k in range(2, 13):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=600, max_iter=100,random_state=0)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_coefficients.append(score)
#close("all") 
#plt.set_size_inches(18.5, 10.5)
plt.style.use("fivethirtyeight")
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')

plt.plot(range(2, 13), silhouette_coefficients)
plt.xticks(range(2, 13))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
#plt.figure(figsize=(3,3))
plt.savefig('Silhouette_Score_lineplot.pdf',dpi=100)
#plt.show()
#=============================================================================