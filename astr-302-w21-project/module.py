#!/usr/bin/env python

from ipywidgets import interact, fixed

import numpy as np

import matplotlib.pyplot as plt

import astropy
from astropy.nddata import CCDData

import photutils
from astropy.stats import sigma_clipped_stats    

def image_reader(filename):
    """Takes in a filename/directory string, and then reads the data using
    astropy's CCDData attribute, where the file is in .fits file format.
    """
    
    image =  CCDData.read(filename, unit = 'adu')
    return image

def plot(image, radius, mul):
    """Use photutils DAO starfinder to get the locations of any
    objects in the image (above) with the threshold being one of the input
    values. Then use astropy's statistics to subtract the median from the image.
    Makes an array of two arrays which are our data points locations, then transpose
    those points to get the actual locations on our .fits image. Uses photutils CircularAperture
    to put an aperture around each potential star in our positions array.
    """
    
    mean, median, std = astropy.stats.sigma_clipped_stats(image.data)
    DAO = photutils.detection.DAOStarFinder(threshold = mul * std, fwhm = 3.0)
    sources = DAO(image.data - median)
    source_points = np.array([sources['xcentroid'], sources['ycentroid']])
    positions = np.transpose(source_points)
    apertures = photutils.CircularAperture(positions, r = radius)
    fig, ax = plt.subplots(figsize = (15, 15))
    ax.imshow(image.data, cmap = 'Greys', origin = 'lower', vmin = 100, vmax = 5000)
    ax.set_title('Interactive Starfinder')
    apertures.plot(color = 'c', lw = 1, alpha = 1)
    
def interactive_apertures(image):
    """Uses ipywidgets to create an interactive graphing experience. Specifically,
    it calls on the interact attribute of ipywidgets, then calls our plot function, 
    and specifies that the image is fixed, while the radius can be interacted with,
    with a range of 1 pixel to 20 pixels. The mul variable stands for a standard deviation
    multiplier for our threshold of relevance, with a range from 0.5 to 25,
    incrementing by 0.5.
    """
    
    interact(plot, image = fixed(image), radius = (1, 20, 1), mul = (0.5, 25, 0.5))
    