# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:41:54 2019

@author: iasedric
"""

import numpy as np
#import matplotlib.pyplot as plt
#import h5py
#import scipy
from PIL import Image
#from scipy import ndimage
#from lr_utils import load_dataset
import os, sys



def resize():
    path = "C:\\Users\\iasedric.REDMOND\\Documents\\_Perso\\Training\\Deep Learning\\train\\"
    dirs = os.listdir( path)
    #print("for")
    for item in dirs:
        print(item)
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((300,300), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)

resize()


def trainset():
    path = "C:\\Users\\iasedric.REDMOND\\Documents\\_Perso\\Training\\Deep Learning\\train\\"
    dirs = os.listdir( path)
    #print("for")
    for item in dirs:
        im = Image.open(path+item)


data=np.array(im.getdata())
data_flat=data.reshape((data.shape[0]*3,1))