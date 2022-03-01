#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 22:22:11 2022

@author: natalia
"""
import iris_lmsalpy.extract_irisL2data as extract_irisL2data

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.wcs import WCS
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import pandas as pd
#import dask
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import constants
from utils import *

def centroid_summary(centroids, rows=14, cols=4, n_bins =89):
    '''
    Inspired by Brandon
    
    plots a summary of the centroids found by the k-means run
    '''
    
    #n_bins = 89
    core_1 = 1334.53
    core_2 = 1335.71
    lambda_min = 1334.0002102
    lambda_max = 1336.2846902
    xax = np.linspace( lambda_min, lambda_max, n_bins )
    
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True)
    
    fig.subplots_adjust(hspace=0.7) 
    ax=axs.ravel()

    
    for k in range(len(centroids)):
        ax[k].plot(xax, centroids[k], color='black', linewidth=1.5, linestyle='-')
        ax[k].axvline(x=core_1,color='black',linewidth = 1)
        ax[k].axvline(x=core_2,color='black',linewidth = 1)
        ax[k].set_xticks([])
        ax[k].set_yticks([])
        ax[k].set_xlim(lambda_min,lambda_max)
        ax[k].set_ylim(0,1)
        ax[k].set_title(f'Group {k}', fontsize=8)
        #ax[k].text( .02, .82, str(k), transform=ax[k].transAxes, size=15)
        
    for j in range(len(centroids), cols*rows):
        ax[j].axis("off")

    plt.show()
    

    return None

def plot_cluster_members(cluster_number, spectra, k_means_labels, k_means_centroids, 
                         wavelength, index_range=None, member_number=None, multiple_clusters_mean=None):
    
    """
    Plotting spectra corresponding to a cluster 
    
    input: cluster_number --> Cluster number from k-means
           spectra --> 2D array containing all CII spectra (from extract_CII_spectra)
           k_means_labels --> labels outputted from k means
           k_means_centroids --> ceontoids outputted from k means
           wavelength --> wavelength range to plot against
           member_number --> number of members to plot 
           multiple_clusters_mean --> if true then need cluster_number to be a list of clusters
                                      mean of the clusters will be plotted
    
    output: plot of spectrum of a cluster group with its assigned spectra
    """
       
    if k_means_labels.ndim == 2:
        k_means_labels = k_means_labels.flatten()
    
    #Only if spectra input is the actual data cube used
    if spectra.ndim==3: 
            spectra = extract_spectra(spectra, index_range=index_range)
    
    
    if multiple_clusters_mean:
        matching_index = np.where(np.isin(k_means_labels, cluster_number))[0]
        
        clusters_total = k_means_centroids[cluster_number[0]]
        for cluster in cluster_number[1:]:
            clusters_total = np.vstack((clusters_total, k_means_centroids[cluster]))
        cluster_mean = np.mean(clusters_total, axis=0)
        
        
    else:
        
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
    
    if multiple_clusters_mean:
        for cluster in cluster_number:
            plt.plot(wavelength, k_means_centroids[cluster], linewidth=1.5, linestyle='-', label=f"Centroid {cluster}")
        
        plt.plot(wavelength, cluster_mean, linewidth=1.5,color='black', linestyle='-', label= "Mean Centroid")
    else:
        plt.plot(wavelength, k_means_centroids[cluster_number], color='black', linewidth=1.5, linestyle='-', label="Assigned Centroid")
    
    plt.axvline(x=133.453,color='black',linewidth = 1)
    plt.axvline(x=133.571,color='black',linewidth = 1)
    plt.legend()
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
    
    #axs[0].imshow(intensities, vmin=0, vmax=45, cmap='hot')
    axs[0].imshow(intensities, vmin=0, vmax=800, cmap='Greys')
    im1 = axs[1].imshow(labels, cmap='gist_ncar')
    plt.gca().invert_yaxis()
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
    labels_copy = np.repeat(labels_copy, repeats=2, axis=1)
    labels_masked = np.ma.masked_where( np.isin(labels_copy, labels_to_plot, invert=True), labels_copy)
    

    color_list = ['darkorange','lime', 'mediumorchid', 'red', 'cornflowerblue', 'yellow', 'cyan']
    colorbar_len = len(labels_to_plot)
    
    labels_max_add = np.max(labels_to_plot) +1 
    bounds = np.append(labels_to_plot, labels_max_add)
    
    plt.figure(figsize=(20,20))
    
    cmap = colors.ListedColormap(color_list[:colorbar_len+1])
    cmap.set_bad(color='none')
    
    norm = matplotlib.colors.BoundaryNorm(bounds, colorbar_len)

    #plt.imshow(intensities, vmin=0, vmax=45, cmap='Greys')
    plt.imshow(intensities, vmin=0, vmax=600, cmap='Greys')
    #plt.imshow(labels_masked, alpha = 0.7,  cmap=cmap, norm=norm)
    plt.imshow(labels_masked, alpha=0.5,  cmap=cmap, norm=norm)
    plt.gca().invert_yaxis()
    
    cb = plt.colorbar(ticks=labels_to_plot)
    cb.ax.set_yticklabels(labels_to_plot)
    
    plt.xlabel('X [pix]', fontsize=15)
    plt.ylabel('Y [pix]', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('SJI of a flare', fontsize=15)

    plt.show()
    
#############   
# This function is for plotting masked labels with ellipses and circles for report figure purposes
#
#############    
def superposition_raster_labels_masked(intensities, labels, labels_to_plot, ):
    """
    Plotting an image along with masked spectral labels on top 
    input: intensities --> 2D list/array of intensities
           labels --> 1D or 2D array of labels
           labels_to_plot --> list of label numbers which to include (i.e which cluster groups to include)
    
    """
    import matplotlib.patches as mpatches
    labels_to_plot = sorted(labels_to_plot)
    
    labels_copy = np.copy(labels)
    labels_masked = np.ma.masked_where( np.isin(labels_copy, labels_to_plot, invert=True), labels_copy)
    

    color_list = ['darkorange','lime', 'mediumorchid', 'red', 'cornflowerblue', 'yellow', 'cyan']
    colorbar_len = len(labels_to_plot)
    
    labels_max_add = np.max(labels_to_plot) +1 
    bounds = np.append(labels_to_plot, labels_max_add)
    
    fig, ax = plt.subplots()
    
    cmap = colors.ListedColormap(color_list[:colorbar_len+1])
    cmap.set_bad(color='none')
    
    norm = matplotlib.colors.BoundaryNorm(bounds, colorbar_len)
    circle_outer = mpatches.Circle(xy=[315, 517], radius=50, facecolor='None', edgecolor='royalblue', linewidth=2,transform=ax.transData)
    circle_inner = mpatches.Ellipse(xy=[320, 508], width=74, height=36, angle=65,
                           facecolor='None', edgecolor='darkred', linewidth=2,transform=ax.transData)
    ellipse= mpatches.Ellipse(xy=[229, 352], width=74*1.5, height=32*2.5, angle=48,
                              facecolor='None', edgecolor='royalblue', linewidth=2,transform=ax.transData)
    ellipse_2= mpatches.Ellipse(xy=[226, 371], width=74, height=36, angle=45,facecolor='None', edgecolor='darkred', linewidth=2,transform=ax.transData)
    #plt.imshow(intensities, vmin=0, vmax=45, cmap='Greys')
    plt.imshow(intensities, vmin=0, vmax=600, cmap='Greys')
    #plt.imshow(labels_masked, alpha = 0.7,  cmap=cmap, norm=norm)
    plt.imshow(labels_masked, alpha=0.3,  cmap=cmap, norm=norm)
    ax.add_patch(ellipse)
    ax.add_patch(ellipse_2)
    ax.add_patch(circle_outer)
    ax.add_patch(circle_inner)
    plt.gca().invert_yaxis()
    
    cb = plt.colorbar(ticks=labels_to_plot)
    cb.ax.set_yticklabels(labels_to_plot)
    
    plt.xlabel('X [pix]', fontsize=15)
    plt.ylabel('Y [pixel]', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('1400 Å SJI image of a connection region between two ribbons', fontsize=15)

    plt.show()
