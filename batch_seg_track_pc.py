#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:26:32 2020

@author: xies
"""

# Import utilities
import sys
import numpy as np
import pandas as pd
from os import path
from skimage import exposure,io
from glob import glob
from re import findall

import h5py

# Suppress warning (sklearn forces warnings)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Load CNN stuff
# Assume script is launched within YeaZ-GUI
sys.path.append('/Users/xies/Desktop/Code/YeaZ-GUI/disk') # in Spyder full path is required bc pathing is weird
sys.path.append('/Users/xies/Desktop/Code/YeaZ-GUI/unet')
import Reader as nd # nd refers to nikon data for legacy reasons
import neural_network as nn
from segment import segment


#%% 

# Enter directory names
dirnames = ['/Users/xies/Desktop/test/']

# Enter subfolder name(s) to iterate through
subdirectories = ['full','pos2','pos3']

# Enter the specific Channel to pick (only one)
channel = 'Phase'

# Glob all filtered tifs in subfolder into dictionary
# Assume: temporal order is reflected in file order (no parsing of frame #)
# Assume: you want to seg+track ALL files in folder
# Assume: there is only one occurence of _tXXX. in the filename for parsing out what the first frame in movie is
#         (useful for subsetting movies)

dirlist = {}
#NB: Opted to use nested dicts since pandas indexing becomes cumbersome for experiments w
# different number of time frames
for dirname in dirnames:
    tiflist = {}
    for subdir in subdirectories:
        print(f'Finding files in {dirname}{subdir}')
        #NB: Windows + Linux has different glob implementation. Windows is case-insensitive, UNIX is case sensitive
        # Test for file extension case by testing either .tif or .TIF        
        lowerlist = sorted(glob(path.join(dirname,subdir,channel,'*.tif')))
        upperlist = sorted(glob(path.join(dirname,subdir,channel,'*.TIF')))
        if len(lowerlist) == 0 and len(upperlist) == 0: # No file found
            print('No files found.')
            break
        if len(lowerlist) > 0 and len(upperlist) == 0: # upper case
            flist = lowerlist
            print(f'{len(flist)} files found with .tif extension')
        if len(lowerlist) == 0 and len(upperlist) > 0: # upper case
            flist = upperlist
            print(f'{len(lowerlist)}Files found with .TIF extension')
        if len(lowerlist) > 0 and len(upperlist) > 0: # Both are found, then likely Windowse
            flist = lowerlist
            print(f'{len(lowerlist)} files found with .tif or .TIF extension. Assuming Windows.')

        tiflist[subdir] = flist
    dirlist[dirname] = tiflist

    # Parse the filename to see which frame is the first file
#    framestr = findall('_t[0-9]+.',flist[0])[0]
#    first_frame_in_folder[subdir] = int(framestr[2:5])

#%% Segment and track indivudal tifs using neural_net predict, segment, and Reader to track and handle outputs

# Setting thresholds manually for now
mask_th = 0.3 # Threshold to generate initial mask
seg_distance = int(3) # Min distance for watershed seeding (has to be int bc this is pixel count)

for (dirname,tiflist) in dirlist.items():
    for (subdir,flist) in tiflist.items():
        
        print('--------')
        print(f'Now working on folder: {subdir}')
        
        # Initialize outputs using their h5 handler
        outputstr = path.join(dirname,subdir,channel)
        hdfpathstr = '' # empty string for no pre-existing data
        newhdfstr = subdir # name of h5 file will be position folder name
        
        reader = nd.Reader(hdfpathstr,newhdfstr,outputstr)
        
        for i,f in enumerate(flist):
            
            # Read and pretreat the image
            im_raw = io.imread(f)
            im_adj = exposure.equalize_adapthist(im_raw) #Run CLAHE
#            im = im_raw*1.0
            im = im_adj.astype(np.float) # cast into float
            
            # CNN predict
            pred = nn.prediction(im, is_pc = True)
            print(pred.sum())
            
            # Threshold prediction to obtain mask
            threshed_mask = nn.threshold(pred, mask_th)
            # Watershed on seeds obtained from mask
            seg = segment(threshed_mask, pred, seg_distance)
            print(f'Segmented file {f}')
            # Save to h5
            reader.SaveMask(i,0, seg)
            temp_mask = reader.CellCorrespondence(i, 0)
            reader.SaveMask(i,0, temp_mask)
            print(f'Tracked file {f}')
            
            
        
        
        
