# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 23:19:08 2019

@author: Violet Saathoff
"""

#Import Libraries
import numpy as np
from Network import *
import os
import gzip
import IDX
import matplotlib.pyplot as plt

#A Class for Loading/Storing the Data
class data:
    """A Class for Loading, Storing, and Manipulating the MNIST Data"""
    #Integer Type Pool
    integers = [int, np.int, np.int16, np.int32, np.int64]
    
    #Data Variables
    x = []
    y = []
    tx = []
    ty = []
    
    #Make Labels
    def getLabels(oldLabels):
        """Vectorize the Labels"""
        labels = []
        for i in oldLabels:
            if i == int(i):
                labels.append(np.zeros((10, 1)))
                labels[-1][int(i)][0] = 1
        
        #Return the Labels
        return labels
    
    #Flatten Images
    def flattenImages(oldImages):
        """Reshape the Images from 2D Images to 1D Vectors"""
        #Flatten the Images
        images = []
        for image in oldImages:
            images.append(np.reshape(image, (np.size(image), 1)))
        
        #Return the Flattened Images
        return images
    
    #Image Normalization Function
    def norm(x):
        """Normalize the Pixels to Floats on the Interval (0, 1)"""
        return np.float32(0.98*x/255 + 0.01)
    
    #Load the MNIST Data
    def load(flatten = True, normalize = True):
        """Load All of the MNIST Data Into the Variables:
            data.x     #Training Images
            data.y     #Training Labels
            data.tx    #Test Images
            data.ty    #Test Labels
        """
        
        #Set the Filepath
        filepath = os.getcwd() + '\\MNIST Data\\'
        
        #Clear the Stored Data
        data.x = []
        data.y = []
        data.tx = []
        data.ty = []
        
        #Load Each Data Set
        names = ['train images.gz', 'train labels.gz', 'test images.gz', 'test labels.gz']
        dataList = []
        for name in names:
            #Read the File
            print('Loading: ' + name)
            dataList.append(IDX.read(filepath + name))
        
        #Normalize the Images
        if normalize == True:
            dataList[0] = data.norm(dataList[0])
            dataList[2] = data.norm(dataList[2])
        
        #Process and Store the Images
        if flatten == True:
            data.x = data.flattenImages(dataList[0])
            data.tx = data.flattenImages(dataList[2])
        else:
            data.x = dataList[0]
            data.tx = dataList[2]
        
        #Process and Store the Labels
        data.y = data.getLabels(dataList[1])
        data.ty = data.getLabels(dataList[3])
        
        #Convet the Data to Lists
        data.x = list(data.x)
        data.y = list(data.y)
        data.tx = list(data.tx)
        data.ty = list(data.ty)

#Linear Network
def newANN(η = 0.003, α = 0.9, λ = 5, γ = 0.85, hidden = [70, 70], name = 'MNIST ANN'):
    """Make a New Model to Train
    
    Parameters
    ----------
    η : float, optional
        The learning rate (generally quite small). 
        The default is 0.003.
    α : float, optional
        Momentum Optimization Parameters. Must be in the interval [0, 1). 
        The default is 0.9.
    λ : float, optional
        L2 Regulatization Parameter. Must be greater than or equal to 0. 
        Typical values are in the interval (0, 10].
        The default is 5.
    γ : float, optional
        Learning Rate Decay. Must be in the interval (0, 1], but should 
        typically be close to 1.
        The default is 0.85.
    hidden : list, optional
        The size of the hidden layers (in order). 
        The default is [70, 70].
    name : string, optional
        The name you wish to give the network. 
        The default is 'MNIST ANN'.

    Returns
    -------
    network : Network
        A neural network which can be trained to recognize handwritten digits
        using the MNIST database.

    """
    #Compute the Layers
    if len(hidden) > 0:
        layers = list(np.hstack((784, hidden, 10)))
    else:
        layers = [784, 10]
    
    #Make the Network (η, α, λ, γ, dropout, delay, verbose)
    network = Network(name, η, α, λ, γ, 0.5, 10, False)
    
    #Add the Layers
    for i in range(len(layers) - 1):
        network += Linear(layers[i], layers[i + 1], [sigmoid, softmax][int(i == len(layers) - 2)])
    
    #Return the Network
    return network

#Convolutional Network
def newCNN():
    pass

#Compute the Accurace of the Network
def accuracy(network):
    """Compute the Accuracy of the Network on the Test Data"""
    #Test Each Test Data
    correct = 0
    N = len(data.tx)
    for k in range(N):
        #Get the Network Activations
        a = network(data.tx[k]).T[0]
        
        #Find the Index of Maximum Activation
        Max = -np.inf
        j = -1
        for i in range(10):
            if a[i] > Max:
                j = i
                Max = a[i]
        
        #Record if the Network Correctly Labeled the Image
        correct += int(data.ty[k][j][0] == 1)
    
    #Return the Accuracy
    return correct/N

#Compute the Accuracy on the Training Data (To Help Check for Overfitting)
def trainAccuracy(network):
    """Compute the Accuracy on the Training Data
    
    While this might help you check for overfitting, it should not be used 
    to evlauate your models, since that can easilly cause you to end up 
    overfitting your data.
    """
    #Test Each Test Data
    correct = 0
    N = len(data.x)
    for k in range(N):
        #Get the Network Activations
        a = network(data.x[k]).T[0]
        
        #Find the Index of Maximum Activation
        Max = -np.inf
        j = -1
        for i in range(10):
            if a[i] > Max:
                j = i
                Max = a[i]
        
        #Record if the Network Correctly Labeled the Image
        correct += int(data.y[k][j][0] == 1)
    
    #Return the Accuracy
    return correct/N

#Train a Network
def train(network, epochs:int = 5, batchsize:int = 100, testPortion:int = 500):
    """Train the Network
    
    Parameters
    ----------
    network : Network
        The neural network you wish to train.
    epochs : int, optional
        The number times you want the model to train from the dataset. 
        The default is 5.
    batchsize : int, optional
        The mini-batch size you wish to use (how many images does the model 
        look at before updating itself). The default is 100.
    testPortion : int, optional
        The number of test images you want the model to look at when 
        printing status updates (more images will give you a more accurate 
        estimate of how well the model is doing, fewer images will allow 
        the training to run faster). 
        The default is 500.

    """
    
    #Train the Network
    network.train(data.x, data.y, data.tx[:testPortion], data.ty[:testPortion], epochs, batchsize)
    
    #Compute/Print the Accuracy
    print('\nAccuracy = %s' % round(100*accuracy(network), 2) + '%')
    
    #Plot the Network's Training Curve
    network.plot()