#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General functions used to load IRIS data, perform machine learning and plotting
"""
import iris_lmsalpy.extract_irisL2data as extract_irisL2data

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.wcs import WCS
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
#import dask
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_IRIS_data(file_name: str, spectral_window:str , wavelength_min = None, 
                   wavelength_max = None) -> tuple:
    """
    Loading in raster spectra from IRIS
    
    parameters: filename --> Absolute path to fits file
                spectral_window --> Spectral window to load in
                wavelength_min --> Mnimum wavelength if restrecting wavelength range
                wavelength_max --> Maximum wavelength if restrecting wavelength range

    return: data --> data cube with all spectral and image data
            wavenlength --> wavelength range for CII lines
            index_range --> range to use to limit spectral range (boolean array) equivalent to the limited
            sp --> IRIS class, can be used to plot interactive spectra with sp.quick_look()
    """
    
    raster_filename = file_name
    
    
    
    try:
        #C II 1336
        sp = extract_irisL2data.load(raster_filename,window_info=[spectral_window],memmap=True)
        
    except Exception as e:
        print(e)
        
    data = sp.raster[spectral_window].data
    hd = extract_irisL2data.only_header(raster_filename,extension=1)

    m_to_nm = 1e9
    hd['CDELT3'] = 1.e-9  # Fix WCS for sit-and-stare
    wcs = WCS(hd)

    nwave= data.shape[2]

    wavelength = wcs.all_pix2world(np.arange(nwave), [0.], [0.], 0)[0] * m_to_nm
    
    if wavelength_min or wavelength_max:
        
        if not wavelength_min:
            wavelength_min = np.min(wavelength)
            
        elif not wavelength_max:
            wavelength_max = np.max(wavelength)
            
        else:
            pass
        #133.4 to 133.63
        index_range = (wavelength>wavelength_min)*(wavelength<wavelength_max)

        wavelength = wavelength[index_range]
    
        return data, index_range, wavelength, sp
    
    else:
        
        return data, wavelength, sp



def extract_spectra(data_cube: np.ndarray, index_range=None) -> np.ndarray:
    """
    Extracts all the spectra from a datacube and normalises them 
    
    parameters: data_cube --> data cube with all spectra (from load_IRIS_data)
                index_range --> range to use to limit soectral range (boolean array)
    
    return: spectra --> two dimensional array. Each row corresponds to a CII spectrum
    """
    print('Extracting spectra....')
    
    spectra_flatten = data_cube.reshape(-1, data_cube.shape[2])
    
    if index_range is not None:
        spectra_flatten = spectra_flatten[:, index_range]
        
    spectra_max = spectra_flatten.max(axis=1)
    spectra_normalized = spectra_flatten/spectra_max[:, np.newaxis]
    
    return spectra_normalized



def mini_batch_k_means(X, n_clusters=10, batch_size=10, n_init=10, verbose=0):

    '''
   Credits to Brandon
    Return centroids and labels of a data set using the k-means algorithm in minibatch style

    input: X --> dataset
           n_clusters --> number of groups to find in the data |k|
           batch_size --> number of data points used for a single update step of the centroids
           n_init --> number of times to run k-means, the iteration with the lowest inertia is retained
           verbose --> output statistics
           init --> k-means++ or random

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

    return centroids, labels, inertia, n_clusters, mbk


def centroid_summary(centroids, rows=14, cols=4 ):
    '''
    Inspired by Brandon
    
    plots a summary of the centroids found by the k-means run
    '''
    
    n_bins = 89
    core_1 = 1334.55
    core_2 = 1335.58
    lambda_min = 1334
    lambda_max = 1336.6
    xax = np.linspace( lambda_min, lambda_max, n_bins )
    
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(8.27,16))
    
    fig.subplots_adjust(hspace=0.9) 
    ax=axs.ravel()

    
    for k in range(len(centroids)):
        ax[k].plot(xax, centroids[k], color='black', linewidth=1.5, linestyle='-')
        #ax[k].axvline(x=core_1,color='black',linewidth = 1)
        #ax[k].axvline(x=core_2,color='black',linewidth = 1)
        #ax[k].set_xticks([])
        #ax[k].set_yticks([])
        ax[k].set_xlim(lambda_min,lambda_max)
        ax[k].set_ylim(0,1)
        ax[k].set_title(f'Group number {k}', fontsize=8)
        #ax[k].text( .02, .82, str(k), transform=ax[k].transAxes, size=15)
        
    for j in range(len(centroids), cols*rows):
        ax[j].axis("off")

    plt.show()
    #plt.savefig('/Users/natalia/Desktop/centroid_summary.png', dpi=300)
    

    return None


#TODO figure out how to speed up silhouette method. Don't use for now!
def Kmeans_optimisation(spectra, max_clusters):
    """
    Get silhouette score and inertia to evaluate ideal number of cluster members
    
    input: spectra --> 2 dimensional array containing all spectral lines used from k-means.
           k_meas_data --> data outputted from mini_batch_k_means
    
    output: intertias --> 1D array containing interia for different cluster numbers.
            silhouette --> 1D array containing silhouette score for different cluster numbers.
    """
    
    inertias = []
    silhouette = []
    for i in range(2, max_clusters):
        ml_data = mini_batch_k_means(spectra, n_clusters=i, batch_size = 200, n_init=100)
        #score = dask.delayed(silhouette_score)(spectra,ml_data[1], metric='euclidean')
        inertias.append(ml_data[2])
        silhouette.append(score)
    
    #inertias = dask.compute(*inertias)
    #silhouette = dask.compute(*silhouette)
    return inertias, silhouette



def plot_cluster_members(cluster_number, spectra, k_means_labels, k_means_centroids, 
                         wavelength, index_range=None, member_number=None):
    
    """
    Plotting spectra corresponding to a cluster 
    
    input: cluster_number --> Cluster number from k-means
           spectra --> 2D array containing all CII spectra (from extract_CII_spectra)
           k_means_labels --> labels outputted from k means
           k_means_centroids --> ceontoids outputted from k means
           wavelength --> wavelength range to plot against
           member_number --> number of members to plot 
           
    
    output: plot of spectrum of a cluster group with its assigned spectra
    """
       
    if k_means_labels.ndim == 2:
        k_means_labels = k_means_labels.flatten()
    
    #Only if spectra input is the actual data cube used
    if spectra.ndim==3: 
            spectra = extract_spectra(spectra, index_range=index_range)
        
    matching_index = np.where(k_means_labels==cluster_number)[0]
    
    if member_number is None:
        member_number = len(matching_index)
    
    plt.figure(figsize=(9, 6))
    plt.ylim(-0.2, 1)
    plt.title(f"Cluster number {cluster_number}")
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Normalised intensity [DN]')
        
    for i in range(member_number):
            plt.plot(wavelength, spectra[matching_index[i]], color='gray', linewidth=1, linestyle='dotted')
            
       
    plt.plot(wavelength, k_means_centroids[cluster_number], color='black', linewidth=1.5, linestyle='-')
    plt.show()

    
    return None



def plot_raster(data_cube, intensity_wavelength):
    """
    Plotting the actual image
    
    input: data_cube --> data cube outputted from load_iris 
           intensity_wavelength --> wavelength index at which to view the image.
    
    """
    plt.figure(figsize=(15, 15))
    plt.imshow(data_cube[..., intensity_wavelength], vmin=0, vmax=45, cmap='hot')
    plt.title(f'X-Y image at wavelength {intensity_wavelength}')
    plt.show()


def plot_image_labels(intensities, labels):
    """
    Plotting an image along with spectral labels assigned to each pixel
    input: intensities --> 2D list/array of intensities
           labels --> 1D or 2D array of labels
           
    """
    if labels.ndim == 1:
        labels = labels.reshape(intensities.shape[0], intensities.shape[1])
        
    fig, axs = plt.subplots(1, 2, figsize=(20, 20), sharey = True)
    plt.subplots_adjust(wspace=.0)
    
    axs[0].imshow(intensities, vmin=0, vmax=45, cmap='hot')
    im1 = axs[1].imshow(labels, cmap='gist_ncar')
    
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')
    
    plt.show()


def superposition_raster_labels(intensities, labels, labels_to_plot):
    """
    Plotting an image along wth spectral labels on top
    input: intensities --> 2D list/array of intensities
           labels --> 1D or 2D array of labels
           labels_to_plot --> list of label numbers which to include (i.e which cluster groups to include)
    
    """
    if labels.ndim == 1:
        labels = labels.reshape(intensities.shape[0], intensities.shape[1])
        
    labels_to_plot = sorted(labels_to_plot)
    
    labels_copy = np.copy(labels)
    labels_masked = np.ma.masked_where( np.isin(labels_copy, labels_to_plot, invert=True), labels_copy)
    

    color_list = ['darkorange','lime', 'mediumorchid', 'red', 'cornflowerblue', 'yellow', 'cyan']
    colorbar_len = len(labels_to_plot)
    
    labels_max_add = np.max(labels_to_plot) +1 
    bounds = np.append(labels_to_plot, labels_max_add)
    
    plt.figure(figsize=(20,20))
    
    cmap = colors.ListedColormap(color_list[:colorbar_len+1])
    cmap.set_bad(color='none')
    norm = matplotlib.colors.BoundaryNorm(bounds, colorbar_len)

    plt.imshow(intensities, vmin=0, vmax=45, cmap='Greys')
    plt.imshow(labels_masked, alpha = 0.5,  cmap=cmap, norm=norm)
    
    cb = plt.colorbar(ticks=labels_to_plot)
    cb.ax.set_yticklabels(labels_to_plot)
    
    plt.xlabel('X (pixel)')
    plt.ylabel('Y (pixel)')

    plt.show()
    
    
# Ignore these functions
def change_cluster(centroids, cluster_delete, cluster_add, labels):
    labels = np.array(labels)
    centroids = np.delete(centroids, [cluster_delete], axis=0)
    labels[labels==cluster_delete] = cluster_add
    
    if cluster_delete<len(centroids):
        for i in range(cluster_delete+1, len(centroids)+1):
            labels[labels==i] = i-1
            
    return centroids, labels

def interpolate_correct_labels(labels, merge_cluster, wrong_clusters:list, centroids, raster_size:tuple):
    #[55, 8, 7, 3, 2] 1
    """
    Used for merging and deleting clusters along with reshaping a 1D array of labels into a 2D array that can be used for plotting
           
    """
    
    wrong_clusters = sorted(wrong_clusters, reverse=True)
    correct_labels = np.array([])
    correct_labels = change_cluster(centroids, wrong_clusters[0], 1, labels)[1]
    for cluster in wrong_clusters[1:]:
        #centoid = k_means_tot_f[0]
        correct_labels = change_cluster(centroids, cluster, 1, correct_labels)[1]
    
    #rotating the label 2D distribution, raster_size = (y, x)
    correct_labels = np.reshape(correct_labels, raster_size)
    correct_labels = np.rot90(correct_labels)
    correct_labels = np.flipud(correct_labels)
    
    return correct_labels



