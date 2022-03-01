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
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
#import dask
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import constants
import pywt
from scipy import signal
from scipy.optimize import curve_fit
from iris_plot import *

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
    
        return data, index_range, wavelength, sp, wcs
    
    else:
        
        return data, wavelength, sp, wcs



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



def mini_batch_k_means(X, n_clusters=10, batch_size=10, n_init=10, verbose=0, init='k-means++'):

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
                          n_init=n_init, max_no_improvement=1, verbose=verbose)

    mbk.fit(X)

    centroids = mbk.cluster_centers_
    labels = mbk.labels_
    inertia = mbk.inertia_

    return centroids, labels, inertia, n_clusters, mbk

def KNN(x_train, labels_train, n_neighbors=1, x_test):
    
    k_nn = KNeighborsClassifier(n_neighbors=n_neighbors)
    k_nn.fit(x_train, labels_train)
    
    labels_test = k_nn.predict(x_test)
    
    return labels_test


def merge_centroids(clusters_delete, k, centroid_list_update):
    
    clusters_delete = np.sort(clusters_delete)[::-1]
    clusters_mean = k[0][clusters_delete[0]]
    clusters_mean = np.vstack((clusters_mean, k[0][clusters_delete[1]]))
    clusters_mean = np.mean(clusters_mean, axis=0)
    
    centroid_list_update = np.delete(centroid_list_update, clusters_delete[0], 0)
    centroid_list_update = np.delete(centroid_list_update, clusters_delete[1], 0)
    
    centroid_list_update = np.append(centroid_list_update, [clusters_mean], axis=0)
    return centroid_list_update
    
    
# Ignore this functions
def change_cluster(centroids, cluster_delete, cluster_add, labels):
    """
    used to replace a cluster with different one     
    """
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



def smoothing(spectra, wavelength=None, plot=False):
    print("Smoothing spectra...")
    
    spectra_resampled = signal.resample(spectra, 512, axis=1)
    for i in range(len(spectra_resampled)):
        if not i % 1000:
            print(i)
        wavelets = pywt.swt(spectra_resampled[i], 'coif2')
        wavelets[1:] = (pywt.threshold(ii, value=0.2, mode="soft" ) for ii in wavelets[1:])
        wavelets_reconstructed = pywt.iswt(wavelets, 'coif2')
        
        #normalise
        max_intensity = np.max(wavelets_reconstructed)
        wavelets_reconstructed = wavelets_reconstructed / max_intensity
        spectra_resampled[i] = wavelets_reconstructed
        
        if plot:
            plt.figure()
            plt.plot(wavelength, spectra)
            #plt.plot(np.linspace(np.min(wavelength), np.max(wavelength), num=512), wavelet_cut, label='Cut 5')
            plt.plot(np.linspace(np.min(wavelength), np.max(wavelength), num=512), wavelets_reconstructed, label='Tresholding 0.2')
            plt.xlabel('Wavelength')
            plt.ylabel('Normalized intensity')
            plt.title('Original data points vs removed fluctuations with coif2 wavelets')
            plt.legend()
    
    return spectra_resampled
    


def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
    return popt

def fwhm_intensity(spectrum, wavelength, peak_1_range, peak_2_range):
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the datasset.  The function
    uses a spline interpolation of order k.
    """
    
    fwhm = []
    intensity = []
    for peak_range in [peak_1_range, peak_2_range]:
        intensity_max = np.max(spectrum[peak_range[0]:peak_range[1]])
        H, A, x0, sigma = gauss_fit(wavelength[peak_range[0]:peak_range[1]], 
                                    spectrum[peak_range[0]:peak_range[1]])
        FWHM = 2.35482 * sigma
        fwhm.append(FWHM)
        
        intensity.append(intensity_max)

    return fwhm, intensity



class Centroid:
    def __init__(self, centroid_number, centroid_spectrum, fwhm, intensity_max):
        self.centroid_number = centroid_number
        self.centroid_spectrum = centroid_spectrum
        self.fwhm_left = fwhm[0]
        self.fwhm_right = fwhm[1]
        self.intensity_left = intensity_max[0]
        self.intensity_right = intensity_max[1]
        self.intensity_ratio = np.min(intensity_max) / np.max(intensity_max)

def centroid_analysis(centroid_number, n_clusters, centroids, labels, spectra, wavelength):
    centroid = centroids[centroid_number]
    labels_index = np.where(labels == centroid_number)[0]
    spectra_cluster = spectra[labels_index]
    
    print(len(spectra_cluster))
    k_means = mini_batch_k_means(spectra_cluster, n_clusters=n_clusters, batch_size=len(spectra_cluster))
    
    centroids_info = []
    for i in range(len(k_means[0])):
        centr = k_means[0][i]
        fwhm, intensity_max = fwhm_intensity(centr, k=3)
        c = Centroid(i, centr, fwhm, intensity_max)
        centroids_info.append(c)
    
    similar_spectra_plot(centroids_info, wavelength, centroid_number)
    
    for ii in range(len(k_means[0])):
        plot_cluster_members(ii, spectra_cluster, k_means[1], k_means[0], wavelength)
        
    return k_means[0] 




def bisector_velocity(spectrum, wavelength, interpolation_number_rows, line_side):
    """
    Function for performing bisecot analysis of two spectral lines
    """
    intensity_max_index = np.where(spectrum==np.max(spectrum))[0][0]
    
    left_side_df = pd.Series(wavelength[0:intensity_max_index], index=spectrum[0:intensity_max_index])
    right_side_df = pd.Series(wavelength[intensity_max_index:], index=spectrum[intensity_max_index:])
    
    interpolate_rows = right_side_df.iloc[1::interpolation_number_rows].index
    nans = np.zeros(len(interpolate_rows)) + np.nan
    
    interpolate_rows_df = pd.Series(nans, index=interpolate_rows)
    
    left_side_df = left_side_df.append(interpolate_rows_df)
    
    left_side_df = left_side_df.sort_index(ascending=False)
    
    left_side_df = left_side_df.interpolate(method='index')
    
    left_side_df_reverse = pd.Series(left_side_df.index, index=left_side_df.values)
    right_side_df_reverse = pd.Series(right_side_df.index, index=right_side_df.values)
    
    wavelengths_left = left_side_df_reverse[left_side_df_reverse.isin(interpolate_rows[1:])].index
    wavelengths_right = right_side_df_reverse[right_side_df_reverse.isin(interpolate_rows[1:])].index

    wavelength_df = pd.DataFrame({'intensity': interpolate_rows[1:], 'wavelengths_left':wavelengths_left, 'wavelengths_right':wavelengths_right})
    
    wavelength_df['mean_wavelength'] = wavelength_df[['wavelengths_left', 'wavelengths_right']].mean(axis=1)
    
    if line_side == 'left':
        rest_wavelength = 133.4532
    else:
        rest_wavelength = 133.5708
        
    velocities = (wavelength_df['mean_wavelength'] - rest_wavelength) * constants.c / rest_wavelength 
    

    wavelength_df['velocity'] = velocities 
    
    return (wavelength_df)
    
def mask_ellipse(center_x, center_y, width, height, angle, array_size):
    """
    Crease a mask for an array
    
    array_size = [602, 436]
    """
    from skimage.draw import ellipse
    
    mask = np.zeros((array_size[0], array_size[1]), dtype=np.uint8)
    rr, cc = ellipse(center_y, center_x, height, width, rotation=np.deg2rad(angle))
    mask[rr, cc] = 1
    mask[mask==0] = False
    mask[mask==1] = True
    
    return mask