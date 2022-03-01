# K-means for classification of spectra from IRIS level 2 data

An example of how to run the k-means algorithm on a single data file:

### 1. Load the data


To load the data you need to specify: 
  * Path to the file      
  * The lines you want to extract (in this case CII, check [IRIS level 2 guide](https://iris.lmsal.com/itn45/IRIS-LMSALpy_chapter6.html) for more info)      
  * Optional: Minimum and/or maximum wavelength into which to crop the spectra. This wavelength range will be used to return a boolean array (idex range) that can be used to actually crop the spectra in futher steps. 

```python
 data_cube, index_range, wavelength, sp = load_IRIS_data(f'iris_l2_20140815_070803_3800258196_raster_t000.fits', 
                                                          'C II 1336', wavelength_min=133.4, wavelength_max=133.63)
```
### 2. Prepare spectra for k-means

This includes normalising and flattening the spectra into a 2D array

To prepare the spectra:
  * ddata cube - This can also include a cropped cubed to a specific flare region
  * Optional: index_range - this will be used to crop the spectra if neccessary
```python
spectra = extract_spectra(data_cube, index_range=index_range)
```  
### 3. You can repeat step 1 and 2 for as many files as you wish to include in the k-means algorithm. But, before putting in into the algorithm one needs to stack all the spectra into a single array

### 4. Run k-means

This step trains the k-means model.

To run the function:
   * spectra - the final list of normalised (and cropped) spectra 
   * n_clusters - number of clusters into which to cluster the spectra (this is very tricky to determine)
   * batch_size - the size of array within the whole spectral array to use for k-mans (this is because we are using MiniBatchKmeans to speed up the process)
   * n_init - number of times to repeat the clustering. Usually more initiations = better results
```python  
k_means = mini_batch_k_means(spectra, n_clusters=50, batch_size=40000, n_init=10)
```

### 5. Plot all the cluster groups

This plots a summary of the outputted clusters from k-means

To run the function:
  * Centroids from k-means
```python
centroid_summary(k_means[0], rows=10, cols=5)
```

<img src="/images/centroids.png" width="400" height="500">

### 6. This is the supervised part of the pipline

This step requires to merge or split centroids with high variance. Once the final list of clusters and labels is obtained pass the result to a K-NN algorithm (KNN function) with K=1.

### 7. Classify spectra of a test image

You will need to do step 1 and 2 for a data cube for which you wish to classify your spectra

```python
labels_test = k_means[4].predict(spectra_test)
```
### 8. Plot spectra assigned to each cluster

This will plot the spectral profiles along the the cluster they were assigned to. Useful for analysis

To run you need:
  * Cluster number (in this case 50)
  * Spectra used for classification(can be both 2D and 3D array)
  * Labels
  * Centroids outputted from k-means
  * Wavelength array (outputted from load_IRIS_data function)
  * Optional: Number of spectral members to plot (just because it can get messy with large numbers)

```python
plot_cluster_members(50, spectra_test, labels_test, k_means[0], data_test_wavelength, member_number=50)
```

<img src="/images/cluster_members.png" width="400" height="280">

### 8. Plot the raster image along side with labels assigned spectra correspoding to each pixel

For this I recomment using a more interactive graphics console such as Qt5 

To plot you need:
  * Intensities to plot - data cube
  * Labels from k-means (can be 1-D or 2-d array)
```python
plot_image_labels(data_cube[...,80], labels_test)
```

<img src="/images/raster_labels.png" width="400" height="400">

### 9. Or you can plot superposition of the labels onto the raster image

I suggest plotting only few cluster groups instead of all assigned groups. Inside the function I made a custom discrete colourmap that has only 7 colours. Hence, you can plot max 7 cluster groups (you can modify it for more cluster groups, just beware of colour contrast with darker part of the raster image).

To plot you need:
  * Intensities to plot - data cube
  * Labels from k-means (can be 1-D or 2-d array)
  * List of cluster numbers for which you want the labels to be visible.
  
```python
superposition_raster_labels(data_cube[...,80], labels_test, [46,38, 26])
```
<img src="/images/image_overlap.png" width="400" height="500">
