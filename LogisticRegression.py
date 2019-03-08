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

#path = "C:\\Users\\iasedric.REDMOND\\Documents\\_Perso\\Training\\Deep Learning\\train\\"
#resize(path)


def trainset(path):
    '''
    Create the trainset: extract images from path, unfold them and create the N x M matrix containing the trainset.
    
    Arguments:
    path -- Path to the folder containing images to include in the trainset.
    '''
    dirs = os.listdir(path)
    trainset = np.zeros((300*300*3,1))
    #print("for")
    for item in dirs:
        im = Image.open(path+item)
        data=np.array(im.getdata())
        data_flat=data.reshape((data.shape[0]*3,1))
        #print(data_flat.shape)
        trainset = np.append(trainset, data_flat, axis=1)
        print(trainset.shape)
        
    trainset = np.delete(trainset, (0), axis=1)
    
    return trainset


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

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros(shape=(dim, 1))
    b = 0
    ### END CODE HERE ###

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    
    #print(dw.shape)
    #print(w.shape)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        grads, cost = propagate(w, b, X, Y)
        
        print ("Cost after iteration %i: %f" % (i, cost))
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule
        w = w - learning_rate * dw  # need to broadcast
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        #if print_cost and i % 100 == 0:
        #    print ("Cost after iteration %i: %f" % (i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


## End of functions
    



path = "C:\\Users\\iasedric.REDMOND\\Documents\\_Perso\\Training\\Deep Learning\\train_small\\"
trainset = trainset(path)

# Standardize the trainset
X = trainset / 255

# Creating the Y (1: cat, 0: dog)
cats = np.ones((1,500))
dogs = np.zeros((1,500))

Y = np.append(cats, dogs, axis=1)

# Initializing the weights and biases
dim = 270000
w, b = initialize_with_zeros(dim)


params, grads, costs = optimize(w, b, X, Y, num_iterations= 2000, learning_rate = 0.0001, print_cost = False)
