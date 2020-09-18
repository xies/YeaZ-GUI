#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:26:32 2020

@author: xies
"""

# Import utilities
import sys
import numpy as np
from os import path
from skimage import exposure,io
from glob import glob
from re import findall

import h5py

# Load CNN stuff
# Assume script is laucnhed within YeaZ-GUI
sys.path.append('/Users/xies/Desktop/Code/YeaZ-GUI/disk')
sys.path.append('/Users/xies/Desktop/Code/YeaZ-GUI/unet')
import Reader as nd # nd refers to nikon data, but mostly historical
import neural_network as nn
from segment import segment

# Suppress warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#%%
# Enter directory name
dirname = '/Users/xies/Desktop/test/'

# Enter subfolder name(s) to iterate through
subdirectories = ['pos1','pos2','pos3']

# Enter the specific Channel to pick
channel = 'Phase'

# Glob all filtered tifs in subfolder into dictionary
# Assume: temporal order is reflected in file order (no parsing of frame #)
# Assume: you want to seg+track ALL files in folder
# Assume: there is only one occurence of _tXXX. in the filename for parsing out what the first frame in movie is
#         (useful for subsetting movies)

tiflist = {}
first_frame_in_folder = {}
for subdir in subdirectories:
    flist = sorted(glob(path.join(dirname,subdir,channel,'*.tif')))
    tiflist[subdir] = flist

    # Parse the filename to see which frame is the first file
#    framestr = findall('_t[0-9]+.',flist[0])[0]
#    first_frame_in_folder[subdir] = int(framestr[2:5])

#%% Segment and track indivudal tifs using neural_net predict, segment, and Reader to handle outputs

# Setting thresholds manually for now
mask_th = 0.3 # Threshold to generate initial mask
seg_distance = int(3) # Min distance for watershed seeding (has to be int bc this is pixel count)

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
        im = im_raw.astype(np.float) # cast into float
        
        pred = nn.prediction(im, is_pc = True)
        
        # Threshold prediction to obtain mask
        threshed_mask = nn.threshold(pred, mask_th)
        # Watershed on seeds obtained from mask
        seg = segment(threshed_mask, pred)
        print(f'Segmented file {f}')
        # Save to h5
        reader.SaveMask(i,0, seg)
        temp_mask = reader.CellCorrespondence(i, 0)
        reader.SaveMask(i,0, temp_mask)
        print(f'Tracked file {f}')
        
        
        
        
        