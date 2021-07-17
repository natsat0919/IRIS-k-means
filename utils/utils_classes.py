#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python classes used in utils code
"""
import numpy as np

class Pixel:
        def __init__(self, raster_number, y_dimension, spectrum):
            self.raster_number = raster_number
            self.y_dimension = y_dimension
            #To make dynamical change 89 to dynamical number
            self.spectrum = np.zeros(len(spectrum))
            self.intensity = 0
            self.cluster_label = 0
            
        def add_class_intensity_spectrum(self, intensity, cluster_label, spectrum):
            self.intensity = intensity
            self.cluster_label = cluster_label
            self.spectrum += spectrum