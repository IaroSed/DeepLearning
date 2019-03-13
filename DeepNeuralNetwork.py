# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:24:42 2019

@author: iasedric
"""


from testCases import *
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward

import numpy as np
#import matplotlib.pyplot as plt
#import h5py
#import scipy
from PIL import Image
#from scipy import ndimage
#from lr_utils import load_dataset
import os


def resize(path):
    '''
    Resize the images to 300x300 size.
    
    Arguments:
    path -- Path to the folder containing images to resize.
    
    '''
    dirs = os.listdir(path)
    #print("for")
    for item in dirs:
        print(item)
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((300,300), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)

#path = "C:\\Users\\iasedric.REDMOND\\Documents\\_Perso\\Training\\Deep Learning\\test\\"
#resize(path)


def dataset(path):
    '''
    Create the dataset: extract images from path, unfold them and create the N x M matrix containing the dataset.
    
    Arguments:
    path -- Path to the folder containing images to include in the dataset.
    '''
    dirs = os.listdir(path)
    dataset = np.zeros((300*300*3,1))
    #print("for")
    for item in dirs:
        im = Image.open(path+item)
        data=np.array(im.getdata())
        data_flat=data.reshape((data.shape[0]*3,1))
        #print(data_flat.shape)
        dataset = np.append(dataset, data_flat, axis=1)
        print(dataset.shape)
        
    dataset = np.delete(dataset, (0), axis=1)
    
    return dataset


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + np.exp(-z))
    ### END CODE HERE ###
    
    return s

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    ### START CODE HERE ### (≈ 4 lines of code)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    ### END CODE HERE ###
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters