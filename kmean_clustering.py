#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:38:53 2021
@author: pradeep
"""
#=============================================================================
#                  This script is Calculates the K mean clustering
#=============================================================================
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
import cartopy.crs as ccrs
from sklearn import metrics
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from matplotlib.pyplot import *
from matplotlib.pyplot import figure

#=======================================================================
fn        = '/home/pradeep/Desktop/Dropbox/PAPER2_REVISED4/DATA/KMEAN_INPUT_H_600_15N_22_68_72E_7.6mm.nc'
dset      = xr.open_dataset(fn)
OLR       = dset['KFILTERED_H'][:][:][:]
OLR1      = OLR
time      = dset['time'][:]
lat       = dset['lat'][:]
lon       = dset['lon'][:]
xreshape  = dset['xreshape'][:][:]
#max_iter=1
#--------------------------------------------------------------------------
kmeans             = KMeans(n_clusters=4, init='k-means++', n_init=20,max_iter=20,random_state=0)
scaled_features    = StandardScaler().fit_transform(xreshape)
kmeans.fit(scaled_features)
KMEAN_LABEL        = kmeans.labels_
#---------------------------------------------------------------------------
#===========================================================================
#                  EXTACT INDEX OF THE 
#===========================================================================
INDEX1   = np.where(KMEAN_LABEL==0)
INDEX2   = np.where(KMEAN_LABEL==1)
INDEX3   = np.where(KMEAN_LABEL==2)
INDEX4   = np.where(KMEAN_LABEL==3)

OLR1     = OLR[INDEX1][:][:]
OLR2     = OLR[INDEX2][:][:]
OLR3     = OLR[INDEX3][:][:]
OLR4     = OLR[INDEX4][:][:]

COM1     = OLR1.mean(axis=0)
COM2     = OLR2.mean(axis=0)
COM3     = OLR3.mean(axis=0)
COM4     = OLR4.mean(axis=0)
#=============================FILLED CONTOUR================================
from   mpl_toolkits.basemap import Basemap, cm
import matplotlib.pyplot as plt
from   netCDF4 import Dataset as open_ncfile
import numpy as np
from scipy import misc
from matplotlib.cbook import dedent
import os
os.environ["PROJ_LIB"] = os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter
import scipy.ndimage
from matplotlib.ticker import NullFormatter

var1  = COM1
var2  = COM2
var3  = COM3
var4  = COM4

var1  = ndimage.gaussian_filter(var1, sigma=0.7, order=0)
var2  = ndimage.gaussian_filter(var2, sigma=0.7, order=0)
var3  = ndimage.gaussian_filter(var3, sigma=0.7, order=0)
var4  = ndimage.gaussian_filter(var4, sigma=0.7, order=0)
print((INDEX4))
#============================================================
#                   making contour plot
#============================================================
dpi = 100
fig = plt.figure()
#fig = plt.figure(figsize=(600/dpi, 500/dpi), dpi=dpi)

#-- create map
map = Basemap(projection='cyl',llcrnrlat= 0.,urcrnrlat= 30.,\
              resolution='c',  llcrnrlon=60.,urcrnrlon=100.)
    
    
ax = fig.add_subplot(2,2,1)
#-- draw coastlines, state and country boundaries, edge of map
map.drawcoastlines()
map.drawstates()
map.drawcountries()
#-- create and draw meridians and parallels grid lines
map.drawparallels(np.arange( 0., 30.,8.),labels=[1,0,0,0],fontsize=10)
map.drawmeridians(np.arange(50.,100.,8.),labels=[0,0,0,1],fontsize=10)
#-- convert latitude/longitude values to plot x/y values
x, y = map(*np.meshgrid(lon,lat))
#-- contour levels
clevs = np.arange(-40,40,5)
#-- draw filled contours
cnplot1 = map.contourf(x,y,var1,clevs,cmap=plt.cm.RdBu_r)
cbar1   =  map.colorbar(cnplot1,location='bottom',pad="10%") 
plt.title('C1')
#================================
ax      = fig.add_subplot(2,2,2)

map     = Basemap(projection='cyl',llcrnrlat= 0.,urcrnrlat= 30.,\
              resolution='c',  llcrnrlon=60.,urcrnrlon=100.)
#-- draw coastlines, state and country boundaries, edge of map
map.drawcoastlines()
map.drawstates()
map.drawcountries()
#-- create and draw meridians and parallels grid lines
map.drawparallels(np.arange( 0., 30.,8.),labels=[1,0,0,0],fontsize=10)
map.drawmeridians(np.arange(50.,100.,8.),labels=[0,0,0,1],fontsize=10)
#-- convert latitude/longitude values to plot x/y values
x, y = map(*np.meshgrid(lon,lat))
cnplot2 = map.contourf(x,y,var2,clevs,cmap=plt.cm.RdBu_r)
#-- add colorbar
cbar2 = map.colorbar(cnplot2,location='bottom',pad="10%")      #-- pad: distance between map and colorbar
     #-- pad: distance between map and colorbar
plt.title('C2')
#===============================================     
ax      = fig.add_subplot(2,2,3)
     
#-- draw coastlines, state and country boundaries, edge of map
map.drawcoastlines()
map.drawstates()
map.drawcountries()

#-- create and draw meridians and parallels grid lines
map.drawparallels(np.arange( 0., 30.,8.),labels=[1,0,0,0],fontsize=10)
map.drawmeridians(np.arange(50.,100.,8.),labels=[0,0,0,1],fontsize=10)

#-- convert latitude/longitude values to plot x/y values
x, y = map(*np.meshgrid(lon,lat))

cnplot3 = map.contourf(x,y,var3,clevs,cmap=plt.cm.RdBu_r)

#-- add colorbar
cbar3 = map.colorbar(cnplot2,location='bottom',pad="10%")      #-- pad: distance between map and colorbar
     #-- pad: distance between map and colorbar
          
plt.title('C3')

#====================================================
ax      = fig.add_subplot(2,2,4)
#-- draw coastlines, state and country boundaries, edge of map
map.drawcoastlines()
map.drawstates()
map.drawcountries()

#-- create and draw meridians and parallels grid lines
map.drawparallels(np.arange( 0., 30.,8.),labels=[1,0,0,0],fontsize=10)
map.drawmeridians(np.arange(50.,100.,8.),labels=[0,0,0,1],fontsize=10)

#-- convert latitude/longitude values to plot x/y values
x, y = map(*np.meshgrid(lon,lat))

cnplot4 = map.contourf(x,y,var4,clevs,cmap=plt.cm.RdBu_r)

#-- add colorbar
cbar4 = map.colorbar(cnplot2,location='bottom',pad="10%")      #-- pad: distance between map and colorbar
     #-- pad: distance between map and colorbar
plt.suptitle('xx')
plt.tight_layout()
plt.subplots_adjust(top=0.85, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=-0.21)
#-- maximize and save PNG file
plt.savefig('plot_matplotlib_contour_filled_rect.png', bbox_inches='tight', dpi=dpi)
#======================================================================
#              wrighting data into netcdf file  
#======================================================================
import netCDF4 as nc
import numpy as np

fn     = '/home/pradeep/Desktop/Dropbox/PAPER2_REVISED4/DATA/KMEAN_OUTPUT_OLR_15N_22_68_72E_7.6mm.nc'

ds     = nc.Dataset(fn, 'w', format='NETCDF4')

label  = ds.createDimension('label', len(time))

labels = ds.createVariable('label', 'f4', ('label',))

LABEL  = ds.createVariable('LABEL', 'f4', ('label'))

LABEL.units = 'Unknown'

labels[:]   = np.linspace(0,len(time)-1,len(time))

print('var size before adding data', LABEL.shape)

LABEL[:]       = kmeans.labels_

print('var size after adding second data', LABEL.shape)

ds.close()
print(kmeans.labels_)

#========================================================================
sse = []
import numpy

for k in range(1, 13):
    
    kmeans = KMeans(n_clusters=k,init='k-means++', n_init=20, max_iter=20, random_state=0)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)

xx = np.linspace(1,13,12)
yy = (numpy.array(sse))*10**-6

kneedle = KneeLocator(xx, yy, S=1, curve="convex", direction="decreasing")

fig = plt.figure()

kneedle.plot_knee()
plt.style.use("fivethirtyeight")
#figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
plt.savefig('kneedle.pdf',dpi=80)

kneedle.plot_knee_normalized()

plt.savefig('plot_knee_normalized.pdf',dpi=10)

print(kneedle.elbow)
#======================================================================