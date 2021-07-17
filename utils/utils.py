#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General functions used to load IRIS data, perform machine learning and plotting
"""
import iris_lmsalpy.extract_irisL2data as extract_irisL2data

import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import dask

from utils_classes import Pixel

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
    """
    
    raster_filename = file_name
    #raster_filename = '/Users/natalia/Downloads/iris_l2_20141025_145828_3880106953_raster_t000_r00000.fits'
    
    
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
    
        return data, index_range, wavelength
    
    else:
        
        return data, wavelength

def spectral_live_plot(data_cube, wavelength, 
                                   pixel_height: int, index_range = None):
    
    """
    Plotting animation of spectral change over time for a given pixel height
    
    input: data_cube --> 2D array containing all spectra (from extract_CII_spectra)
           index_range --> range to use to limit soectral range (boolean array)
           wavelength --> wavelength range to plot against
           pixel_height --> y-pixel
           
    
    output: plot of CII spectra change over time for a give pixel height
    """
    spectrum = data_cube[0, pixel_height]
    
    if index_range is not None:
        spectrum = spectrum[index_range]
    

    plt.ion()
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111)
    line1, = ax.plot(wavelength, spectrum)
    ax.set_ylim(-8, 150)


    for i in range(data_cube.shape[-2]):
        spectrum_update = data_cube[i, pixel_height]
        if index_range is not None:
            spectrum_update = spectrum_update[index_range]
        
        if np.mean(spectrum_update)< -100:
            pass
        
        else:
            plt.title(f'Image/Raster number {i}')
            line1.set_xdata(wavelength)
            line1.set_ydata(spectrum_update)
            fig.canvas.draw()
      
            # to flush the GUI events
            fig.canvas.flush_events()
            time.sleep(1)
    
    return None



def extract_spectra(data_cube: np.ndarray, index_range=None) -> np.ndarray:
    """
    Extract only usable lines from all the spectra, removes any saturated spectra or negative spectra 
    
    parameters: data_cube --> data cube with all spectra (from load_IRIS_data)
                index_range --> range to use to limit soectral range (boolean array)
    
    return: spectra --> two dimensional array. Each row corresponds to a CII spectrum
    """
    print('Extracting spectra....')
    #Use array containing 0s instead of empty array to reduce run time
    spectra = np.zeros(shape=[920000, 89])
    removed = 0
    index = 0
    for i in range(data_cube.shape[-2]):
        
        for row in data_cube[:,i,:]:
            if index_range is not None:
                row = row[index_range]
                
            if np.mean(row) <= 0:
                removed += 1
                pass
            elif len(np.where(row >= 16000)[0])>15:
                removed += 1
                pass
            else:
                maximum = np.max(row)
                spectra[index] = row/maximum
                index +=1
            
    #Removing rows with all 0s
    idx = np.argwhere(np.all(spectra[..., :] == 0, axis=1))
    spectra = np.delete(spectra, idx, axis=0)
    
    return spectra



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


def centroid_summary(centroids, rows=14, cols=4 ):
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
        score = dask.delayed(silhouette_score)(spectra,ml_data[1], metric='euclidean')
        inertias.append(ml_data[2])
        silhouette.append(score)
    
    inertias = dask.compute(*inertias)
    silhouette = dask.compute(*silhouette)
    return inertias, silhouette



def plot_cluster_members(cluster_number, spectra, k_means_labels, k_means_centroids, 
                         wavelength, member_number=None):
    
    """
    Plotting cluster members 
    
    input: cluster_number --> Cluster number from k-means
           spectra --> 2D array containing all CII spectra (from extract_CII_spectra)
           k_means_labels --> labels outputted from k means
           k_means_centroids --> ceontoids outputted from k means
           wavelength --> wavelength range to plot against
           member_number --> number of members to plot 
           
    
    output: plot of spectrum of a cluster group with its assigned spectra
    """
    
    
    
    matching_index = np.where(k_means_labels==cluster_number)[0]
    if member_number is None:
        member_number = len(matching_index)
    
    plt.ylim(-0.2, 1)
    for i in range(member_number):
        plt.plot(wavelength, spectra[matching_index[i]], color='gray', linewidth=1, linestyle='dotted')
    
    plt.plot(wavelength, k_means_centroids[cluster_number], color='black', linewidth=1.5, linestyle='-')
    plt.show()
    
    return None



def plot_raster(data_cube, intensity_wavelength):
    """
    Plotting the actual image
    
    input: data_cube --> data cube outputted from load_iris 
           intensity_wavelength --> wavelength at which to view the image. For CII lines use 133
    
    """
    plt.figure(figsize=(15, 15))
    plt.imshow(data_cube[..., intensity_wavelength], vmin=0, vmax=45, cmap='hot')
    plt.title(f'X-Y image at wavelength {intensity_wavelength}')
    plt.show()

            
def get_complete_data(data_cube: np.ndarray, intensity_wavelength: int, k_means_labels, index_range = None, 
                           individual_components = False):
    """
    IMPORTANT: Only use this function if you used extract_spectra function to get array 
    of spectra to input into the k_means algorithm
    
    input: data_cube --> data cube outputted from load_iris 
           intensity_wavelength --> wavelength at which to view intensity. For CII lines use 133
           k_means_labels --> labels assigned to spectra as outputted by k_means
           index_range --> array to slice the spetcra
           idividual_components --> if True it will only output 3 separate arrays with classes, intensities and spectra
           
    output: if individual_components = False --> 2D array with each entry containing object with class number, spectrum and pixel intensity
            To acess values of Pixel object:
                Pixel spectrum label from k-means - Pixel.cluster_label
                Pixel intensity - Pixel.intensity
                Pixel spectrum - Pixel.spectrum
                Pixel Image number - Pixel.raster_number
                Pixel y-coordinate - Pixel.y_dimension
                
            [[Pixel Object], [Pixel Object], .....  ]]
            [[Pixel Object]], [Pixel Object], ..... ]] 
            [        .              .               ]]     ----> Array shape = total_image_number x max_y-coordinate
            [        .              .               ]]
            [        .              .               ]]
            if individual_components = True -->  Tuple containing 3 separate arrays 
    
    """
    print("Only use this function if you used extract_spectra function to get array of spectra to input into the k_means algorithm")
            
    pix_array_shape = data_cube[..., intensity_wavelength].shape
    pixel_data = np.ndarray(pix_array_shape,dtype=object)
    
    for i in range(data_cube.shape[-2]):
        ii = 0
        for row in data_cube[:,i,:]:
            data = row
            if index_range is not None:
                data = row[index_range]
            
            maximum = np.max(row)
            spectrum = row/maximum
            pixel_data[ii, i] = Pixel(i, ii, spectrum)
            
            if np.mean(data) <= 0:
                pixel_data[ii, i].add_class_intensity_spectrum(row[intensity_wavelength],  np.nan, spectrum)
                ii += 1
                pass
            
            elif len(np.where(data >= 16000)[0])>15:
                pixel_data[ii, i].add_class_intensity_spectrum(row[intensity_wavelength], np.nan, spectrum)
                ii += 1
                pass
            
            else:
               
                pixel_data[ii, i].add_class_intensity_spectrum(row[intensity_wavelength], k_means_labels[0], spectrum)
                k_means_labels = np.delete(k_means_labels, 0)
                ii += 1
                
    if individual_components:
        components = vectorize_array(pixel_data)
        return components
    else:
        return pixel_data

def vectorize_array(pixel_data: np.ndarray) -> tuple:
    """
    Vectorising 2D array with objects
    input: pixel_data --> 2D array outputted from get_complete_data
           
    output: Tuple containing 3 separate arrays with labels, spectra and pixel intensities
    """
    
    inten = lambda d: d.intensity
    vinten = np.vectorize(inten)
    intensities = vinten(pixel_data)
    
    label = lambda c: c.cluster_label
    vlabel = np.vectorize(label)
    labels = vlabel(pixel_data)
    
    sp = lambda s: s.spectrum
    vsp = np.vectorize(sp, otypes=[list])
    spectra = vsp(pixel_data)

    return intensities, labels, spectra


def plot_image_labels(intensities, labels):
    """
    input: intensities --> list/array of intensities
           labels --> list/array of labels
           
    """
    fig, axs = plt.subplots(2, 1, figsize=(20, 20))
    axs[0].imshow(intensities, vmin=0, vmax=45, cmap='hot')
    axs[1].imshow(labels, cmap='Dark2')
    plt.show()

