#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General functions used to load IRIS data, perform machine learning and plotting
"""

import numpy as np
import iris_lmsalpy.hcr2fits as hcr2fits
import iris_lmsalpy.extract_irisL2data as extract_irisL2data

import matplotlib.pyplot as plt
from astropy.wcs import WCS

import time
from scipy.fft import fft, ifft

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import dask

@dask.delayed
def load_IRIS_data(file_name):
    """
    Loading in raster spectra from IRIS
    
    parameters: filename --> Absolute path to fits file

    return: data_c --> data cube with all spectra
            x --> wavelength range for CII lines
            cII_range --> range to use to limit soectral range (boolean array)
    """
    
    raster_filename = file_name
    #raster_filename = '/Users/natalia/Downloads/iris_l2_20141025_145828_3880106953_raster_t000_r00000.fits'

    sp = extract_irisL2data.load(raster_filename,window_info=['C II 1336'],memmap=True)

    data_c = sp.raster['C II 1336'].data
    hd_c = extract_irisL2data.only_header(raster_filename,extension=1)

    m_to_nm = 1e9
    hd_c['CDELT3'] = 1.e-9  # Fix WCS for sit-and-stare
    wcs_c = WCS(hd_c)

    nwave= data_c.shape[2]

    wavelength = wcs_c.all_pix2world(np.arange(nwave), [0.], [0.], 0)[0] * m_to_nm

    cII_range = (wavelength>133.4)*(wavelength<133.63)

    x = wavelength[cII_range]
    
    return data_c, cII_range, x


@dask.delayed
def extract_CII_spectra(data_cube, cII_range):
    """
    Extract only CII lines from all the spectra, removes any saturated spectra or negative spectra 
    
    parameters: data_cube --> data cube with all spectra (from load_IRIS_data)
                cII_range --> range to use to limit soectral range (boolean array)
    
    return: spectra --> two dimensional array. Each row corresponds to a CII spectrum
    """
    
    #Use array containing 0s instead of empty array to reduce run time
    spectra = np.zeros(shape=[775200, 89])
    removed = 0
    index = 0
    for i in range(data_cube.shape[-2]):
        print(i)
        print(removed)
        for row in data_cube[:,i,:]:
            data = row
            if np.mean(data) < 0:
                removed += 1
                pass
            elif len(np.where(data >= 16000)[0])>1:
                removed += 1
                pass
            else:
                maximum = np.max(row[cII_range])
                spectra[index] = row[cII_range]/maximum
                index +=1
            
    #Removing rows with all 0s
    idx = np.argwhere(np.all(spectra[..., :] == 0, axis=1))
    spectra = np.delete(spectra, idx, axis=0)
    
    return spectra


@dask.delayed
def centroid_summary( centroids, rows=14, cols=4 ):
    '''
    Credit to Brandon
    
    plots a summary of the centroids found by the k-means run
    '''

    n_bins = 89
    core_1 = 1334.53
    core_2 = 1335.56
    lambda_min = 1334
    lambda_max = 1336.6
    xax = np.linspace( lambda_min, lambda_max, n_bins )

    fig, axs = plt.subplots(rows, cols, figsize = (15, 15) )
    ax=axs.ravel()
    for k in range(len(centroids)):
        ax[k].plot(xax, centroids[k], color='black', linewidth=1.5, linestyle='-')
        ax[k].axvline(x=core_1,color='black',linewidth = 1)
        ax[k].axvline(x=core_2,color='black',linewidth = 1)
        ax[k].set_xticks([])
        ax[k].set_yticks([])
        ax[k].set_xlim(lambda_min,lambda_max)
#         ax[k].set_ylim(0,1)
        ax[k].text( .02, .82, str(k), transform=ax[k].transAxes, size=15)
    plt.show()

    return None


@dask.delayed
def mini_batch_k_means(X, n_clusters=10, batch_size=10, n_init=10, verbose=0):

    '''
    Credit to Brandon
    Return centroids and labels of a data set using the k-means algorithm in minibatch style

    input: X --> dataset
           n_clusters --> number of groups to find in the data |k|
           batch_size --> number of data points used for a single update step of the centroids
           n_init --> number of times to run k-means, the iteration with the lowest inertia is retained
           verbose --> output statistics

    output: centroids --> average point of each group
            labels --> list of assignments indicating which centroid each data point belongs to
            inertia --> measure of performance, the sum of all intercluster distances
    '''
    # k-means++ initiates the centroids in a smart configuration, promoting better convergence results
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,
                          n_init=n_init, max_no_improvement=10, verbose=verbose)

    mbk.fit(X)

    centroids = mbk.cluster_centers_
    labels = mbk.labels_
    inertia = mbk.inertia_

    return centroids, labels, inertia, n_clusters


@dask.delayed
def Kmeans_optimisation(spectra, ml_data):
    """
    Get silhouette score and inertia to evaluate ideal number of cluster members
    
    input: spectra --> 2 dimensional array containing all CII spectral lines used from k-means.
           ml_data --> data outputted from mini_batch_k_means
    
    output: intertias --> 1D array containing interia for different cluster numbers.
            silhouette --> 1D array containing silhouette score for different cluster numbers.
    """
    
    inertias = []
    silhouette = []
    for i in range(1, 80):
        ml_data = mini_batch_k_means(spectra, n_clusters=i, batch_size = 200, n_init=100)
        score = silhouette_score(spectra,ml_data[1], metric='euclidean')
        inertias.append(ml_data[2])
        silhouette.append(score)
    
    return intertias, silhouette


@dask.delayed
def plot_cluster_members(cluster_number, spectra, ml_data, wavelengths):
    
    """
    Get silhouette score and inertia to evaluate ideal number of cluster members
    
    input: cluster_number --> Cluster number from k-means
           all_spectra --> 2D array containing all CII spectra (from extract_CII_spectra)
           ml_data --> data outputted from mini_batch_k_means
           wavelengths --> wavelength wange to plot against
           
    
    output: plot of spectrum of a cluster group with its assigned spectra
    """
    
    plt.figure(2)

    plt.plot(wavelengths, ml_data[0][cluster_number], color='black', linewidth=1.5, linestyle='-')
    matching_index = np.where(ml_data[1]==cluster_number)[0]
    for i in range(5):
        plt.plot(wavelengths, spectra[matching_index[15+i]], color='gray', linewidth=1, linestyle='dotted')

    plt.show()
    
    return None


@dask.delayed
def plot_spectral_change_animation(wavelengths, cII_range, data):
    #Plotting animation
     """
    Get silhouette score and inertia to evaluate ideal number of cluster members
    
    input: data --> 2D array containing all CII spectra (from extract_CII_spectra)
           cII_range --> range to use to limit soectral range (boolean array)
           wavelengths --> wavelength wange to plot against
           
    
    output: plot of CII spectra change over time for a give pixel height
    """
    
    plt.ion()

    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111)
    line1, = ax.plot(wavelengths, data[265, 0][cII_range])
    ax.set_ylim(-8, 150)


    for i in range(70, 320):
        #def plot_lines(i):
        line1.set_xdata(wavelengths)
        line1.set_ydata(data[265, i][cII_range])
        fig.canvas.draw()
      
        # to flush the GUI events
        fig.canvas.flush_events()
        time.sleep(1)
    
    return None



    
