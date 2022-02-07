# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:54:30 2019

My second attempt at writing a neural network class from scratch.

To Create a Network Use:
network = Network(*params)
network += [Layer](*params)
network.addLayer([Layer](*params)) #This is Equivalent to Using +=
(Examples Below)

Notably, the creator methods do not prevent you from chaining layers togeather
with incompatible shapes.

To Get the output of a network, simply call it: y = network(x)

To Train a Network, use the .train method (using the .GD, .SGD, or .MBGD methods will
also work, but is not recommended as these do not check the inputs). The training method
will be selected automatically based on the batchsize. If batchsize == 0, then the
SGD (Stochastic Gradient Descent) method will be used; else if batchsize < len(data)
or batchsize == 'auto', then the MBGD (Mini-Batch Gradient Descent) method will be used;
finally, if batchsize >= len(data) or batchsize == 'all', then the GD (Gradient Descent)
method will be used. Notably, the SGD method chooses data at random, while MBGD with 
batchsize = 1 will pass over all of the data in order.

Hyperparameters
η - Learning Rate - Range: 0 < η - Typical Values: (0, 0.1]
α - Momentum - Range: 0 <= α <= 1 - Typical Value: 0.9
λ - L2 Regularization - Range: 0 <= λ - Typical Values: [0, 5]
γ - Learning Rate Decay Factor - Range: 0 < γ - Typical Values: [0.9, 1]
epochs - The Number of Passes Over All of the Training Data
batchsize - The Batch Size to go Over Between Network Updates (also selects the training method)

Note: I use "loss" functions to mean vector loss functions, while "cost" functions are
single valued loss functions. Specifically, my cost functions are the average loss per
output neuron: cost(a, y) = sum(loss(a, y))/len(y)

Data Formatting
x, y data should be lists of arrays. It is up to the user to ensrure that each array is
the correct shape. The actual training methods, however, zip the x, y data into a list
of tuples of paired (x, y) data. This allows for easy shuffling, batching, and sampling.

Supported Normalization Functions:
    sigmoid
    tanh
    relu
    leakyRelu
    softmax
    linear (none)
    perceptron

Supported Layer Types:
    Linear
    Bilinear (Not a Particularly Useful Type, But Still Interesting)
    Convolutional (2D)
    Reshape (Typically Used To Go from 2D Layers to 1D Layers)

#Example Network 1
network = Network('MNIST ANN')
network += Linear(784, 100, sigmoid)
network += Linear(100, 100, sigmoid)
network += Linear(100, 100, sigmoid)
network += Linear(100, 10, softmax)

Example Network 2
network = Network('MNIST CNN', 0.1, 0.9)
network += Convolutional((5, 5), 1, 16, relu, pad = (2, 2))
network += Maxpool((2, 2), (1, 1))
network += Convolutional((3, 3), 16, 8, relu, pad = (1, 1))
network += Reshape((6271, 1))
network += Linear(6272, 500, sigmoid)
network += Linear(500, 10, softmax)

Example Training Call
network.train(x, y, x_test, y_test, epochs = 10, batchsize = 32)

@author: Violet Saathoff
"""

#Import Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

#Round a Number to a Specific Number of Sig Figs
def sfRound(x, sf):
    """Round x to sf Significant Figures
    
    Example:
        >>> sfRound(0.0734, 2)
        0.073
    """
    return round(x, int(max(sf - 1, 0) - np.floor(np.log10(abs(x)))))

"""Layer Normalization Functions (Including Derivatives, Loss Functions, and Gradients)"""
#Sigmoid Function
class sigmoid:
    """The Sigmoid Normalization Function
    
    While instances of this class behave as you'd expect the sigmoid function 
    to behave, they also have several other methods which are useful when 
    training networks.
    
    Methods
    -------
    __init__(self) : 
        Create an instance of the sigmoid class.
        
    __call__(self, x) : 
        Computes sigmoid(x) = 1/(1 + e^-x) elementwise on x.
        
    derivative(self, x) : 
        Computes d[sigmoid]/dx(x) = sigmoid(x)*(1 - sigmoid(x)) elementwise 
        on x.
    
    loss(self, a, y):
        Computes the cross-entropy loss function associated with sigmoid 
        given the actual output activation (a) and the expected output 
        activation (y). This is done elementwise.
        (y - 1)*ln(1 - a) - y*ln(a)
    
    cost(self, a, y) -> float:
        Sums the outputs of the loss function (useful for evaluating 
        how well the model is doing during training). The result returned 
        is then normalized per output.
        np.sum(self.loss(a, y))/np.size(y)
    
    grad(self, a, y, z):
        Computes the gradient of the cost function. a = sigmoid(z), but z 
        is passed to the function anyway to save computation (since it is 
        already known). This function is incredibly important for 
        backpropagation. It returns (a - y), which is the linear error in 
        the output. The loss function was actually derived specifically to 
        have this gradient.
    """
    
    #Name of the Normalization Function
    name = 'sigmoid'
    
    #Create the Normalization Object
    def __init__(self):
        pass
    
    #Call the Function
    def __call__(self, x):
        """Compute and Return sigmoid(x) = 1/(1 + e^-x) Elementwise"""
        return 1/(1 + np.exp(-x))
    
    #Call the Derivative of the Function
    def derivative(self, x):
        """Compute and Return the Derivative of sigmoid(x) Elementwise"""
        y = self(x)
        return y*(1 - y)
    
    #Call the Loss Function (Sigmoid Cross Entropy)
    def loss(self, a, y):
        """Compute the Loss Function for the Given Activation/Label"""
        return np.nan_to_num((y - 1)*np.log(1 - a) - y*np.log(a))
    
    #Call the Gradient of the Loss Function
    def grad(self, a, y, z):
        """Compute the Gradient of the Cost Function"""
        return a - y
    
    #Call the Cost Function
    def cost(self, a, y):
        """Compute the Cost Function (np.sum(self.loss(a, y))/np.size(y))"""
        return np.sum(self.loss(a, y))/np.size(y)

#Tanh Function
class tanh:
    #Name of the Normalization Function
    name = 'tanh'
    
    #Create the Normalization Object
    def __init__(self):
        pass
    
    #Call the Function
    def __call__(self, x):
        return np.tanh(x)
    
    #Call the Derivative of the Function
    def derivative(self, x):
        return 1 - self(x)**2
    
    #Call the Loss Function (tanh Cross Entropy)
    def loss(self, a, y):
        return np.nan_to_num((y - 1)*np.log(1 - a) - (y + 1)*np.log(1 + a))/2
    
    #Call the Gradient of the Loss Function
    def grad(self, a, y, z):
        return a - y
    
    #Call the Cost Function
    def cost(self, a, y):
        return np.sum(self.loss(a, y))/np.size(y)

#ReLU Function
class relu:
    #Name of the Normalization Function
    name = 'relu'
    
    #Create the Normalization Object
    def __init__(self):
        pass
    
    #Call the Function
    def __call__(self, x):
        return (x + np.abs(x))/2
    
    #Call the Derivative of the Function
    def derivative(self, x):
        return (np.sign(x) + 1)/2
    
    #Call the Loss Function
    def loss(self, a, y):
        return 0.5*(a - y)**2
    
    #Call the Gradient of the Loss Function
    def grad(self, a, y, z):
        return a - y
    
    #Call the Cost Function
    def cost(self, a, y):
        return np.sum(self.loss(a, y))/np.size(y)

#Leaky ReLU Function
class leakyRelu:
    #Name of the Normalization Function
    name = 'leakyRelu'
    
    #Create the Normalization Object
    def __init__(self, rate = 0.05):
        #Set the Leaky Rate
        self.rate = rate
    
    #Call the Function
    def __call__(self, x):
        return np.sign(x)*np.max([-self.rate*x, x], 0)
    
    #Call the Derivative of the Function
    def derivative(self, x):
        return (1 + self.rate + (1 - self.rate)*np.sign(x))/2
    
    #Call the Loss Function
    def loss(self, a, y):
        return 0.5*(a - y)**2
    
    #Call the Gradient of the Loss Function
    def grad(self, a, y, z):
        return (a - y)*self.derivative(z)
    
    #Call the Cost Function
    def cost(self, a, y):
        return np.sum(self.loss(a, y))/np.size(y)

#Softmax Function
class softmax:
    #Name of the Normalization Function
    name = 'softmax'
    
    #Create the Normalization Object
    def __init__(self):
        pass
    
    #Call the Function
    def __call__(self, x):
        x = np.exp(x - np.max(x))
        return x/np.sum(x, axis = 0)
    
    #Call the Derivative of the Function
    def derivative(self, x):
        raise NotImplementedError('Softmax Derivative not Implemented')
    
    #Call the Loss Function (Softmax Cross Entropy)
    def loss(self, a, y):
        return np.nan_to_num(-y*np.log(a))
    
    #Call the Gradient of the Loss Function
    def grad(self, a, y, z):
        return a - y
    
    #Call the Cost Function
    def cost(self, a, y):
        return np.sum(self.loss(a, y))/np.size(y)

#No Activation Function (Not Recomended for Hidden Layers as that will Reduce the Power of the Network)
class linear:
    #Name of the Normalization Function
    name = 'linear'
    
    #Create the Normalization Object
    def __init__(self):
        pass
    
    #Call the Function
    def __call__(self, x):
        return x
    
    #Call the Derivative of the Function
    def derivative(self, x):
        return np.ones(np.shape(x))
    
    #Call the Loss Function
    def loss(self, a, y):
        return 0.5*(a - y)**2
    
    #Call the Gradient of the Loss Function
    def grad(self, a, y, z):
        return a - y
    
    #Call the Cost Function
    def cost(self, a, y):
        return np.sum(self.loss(a, y))/np.size(y)

#Perceptron Function (Nor Recomended for Use in Actual Networks)
class perceptron:
    #Name of the Normalization Function
    name = 'perceptron'
    
    #Create the Normalization Object
    def __init__(self):
        pass
    
    #Call the Function
    def __call__(self, x):
        return (np.sign(x) + 1) // 2
    
    #Call the Derivative of the Function (Fake Derivative to Facilitate Training)
    def derivative(self, x):
        return np.ones(np.shape(x))
    
    #Call the Loss Function
    def loss(self, a, y):
        return 0.5*(a - y)**2
    
    #Call the Gradient of the Loss Function
    def grad(self, a, y, z):
        return a - y
    
    #Call the Cost Function
    def cost(self, a, y):
        return np.sum(self.loss(a, y))/np.size(y)

"""CLasses for Each Layer Type"""
#Linear Layer
class Linear:
    """Initialize a Feedforward Layer
    
    Parameters
    ----------
    In : int
        The length of the input vector.
    Out : int
        The length of the output vector.
    norm : type
        One of the normalization function classes. Current 
        supported normalization classes are:
            sigmoid
            softmax (Output Layers Only)
            tanh
            relu
            leakyRelu
            linear (Output Layers Only)
            perceptron (Not Recommended)

    Returns
    -------
    Linear : 
        The linear network layer, initialize with random weights/biases.

    """
    #Name of the Layer
    name = 'Linear'
    
    #Create the Layer
    def __init__(self, In:int, Out:int, norm):
        """Initialize a Feedforward Layer
        
        Parameters
        ----------
        In : int
            The length of the input vector.
        Out : int
            The length of the output vector.
        norm : type
            One of the normalization function classes. Current 
            supported normalization classes are:
                sigmoid
                softmax
                tanh
                relu
                leakyRelu
                linear (Output Layers Only)
                perceptron (Not Recommended)

        Returns
        -------
        Linear : 
            The linear network layer, initialize with random weights/biases.

        """
        
        #Check the Input Size
        if type(In) not in Network.integers:
            raise TypeError('In must be an integer greater than 0')
        elif In < 1:
            raise ValueError('In must be an integer greater than 0')
        
        #Check the Output Size
        if type(Out) not in Network.integers:
            raise TypeError('Out must be an integer greater than 0')
        elif Out < 1:
            raise ValueError('Out must be an integer greater than 0')
        
        #Check the Normilization Function
        if norm not in Network.norms:
            raise TypeError('Invalid normalization function provided:' + str(norm))
        
        #Initialize the Weights and Biases
        self.w = np.float32(np.random.randn(Out, In))/np.sqrt(In)
        self.b = np.float32(np.random.randn(Out, 1))
        
        #Set the Normalization Function
        self.norm = norm()
        
        #Save the Input/Output Sizes
        self.In = In
        self.Out = Out
        
        #Initialize the Hyperparameters (These Are Set by the Network Object)
        self.α = 0
        self.η = 0
        self.λ = 0
        self.γ = 0
        self.dropout = 0
        
        #Initialize Arrays for Backpropagation
        self.x = np.zeros((In, 1), dtype = np.float32)
        self.a = np.zeros((Out, 1), dtype = np.float32)
        self.z = np.zeros((Out, 1), dtype = np.float32)
        self.dw = np.zeros((Out, In), dtype = np.float32)
        self.db = np.zeros((Out, 1), dtype = np.float32)
        self.wDrop = np.ones((Out, In))
        self.bDrop = np.ones((Out, 1))
    
    #Copy the Layer
    def copy(self):
        #Create a New Layer
        new = Linear(self.In, self.Out, type(self.norm))
        
        #Copy the Weights/Biases
        new.w = np.copy(self.w)
        new.b = np.copy(self.b)
        
        #Copy the Hyperparameters
        new.α = self.α
        new.η = self.η
        new.λ = self.λ
        new.γ = self.γ
        new.dropout = self.dropout
        
        #Copy the Backpropagation Arrays
        new.x = np.copy(self.x)
        new.a = np.copy(self.a)
        new.z = np.copy(self.z)
        new.dw = np.copy(self.dw)
        new.db = np.copy(self.db)
        
        #Return the New Layer
        return new
    
    #Mutate the Layer
    def mutate(self, rate):
        #Compute the Number of Parameters to Mutate
        N = max(int(round(rate*np.size(self.w))), int(rate*np.size(self.w) > np.random.random()))
        M = max(int(round(rate*np.size(self.b))), int(rate*np.size(self.b) > np.random.random()))
        
        #Mutate N Weights
        for k in range(N):
            #Select a Random Weight
            i = np.random.randint(0, self.Out)
            j = np.random.randint(0, self.In)
            
            #Mutate that Weight
            self.w[i][j] += np.float32(2*self.η*(np.random.random() - 0.5))
        
        #Mutate M Biases
        for k in range(M):
            #Select a Random Bias
            i = np.random.randint(0, self.Out)
            
            #Mutate that Bias
            self.b[i] += np.float32(2*self.η*(np.random.random() - 0.5))
    
    #Compute the Number of Free Parameters in the Layer
    def size(self):
        return np.size(self.w) + np.size(self.b)
    
    #Forward Propagate an Input
    def __call__(self, x):
        return self.norm(np.dot(self.w, x) + self.b)
    
    #Forward Propagate an Input (Saving the Activation and Layers)
    def forward(self, x):
        #Save the Input
        self.x = np.copy(x)
        
        #Get and Save the Activation
        self.z = np.dot(self.w, x) + self.b
        
        #Get and Save the Output Layer
        self.a = self.norm(self.z)
        
        #Return the Output
        return self.a
    
    #Back-Propagate Data
    def backprop(self, δ):
        #Update the Δw Matrix
        self.dw -= np.dot(self.x, δ.T).T
        
        #Update the Δb Vector
        #self.db -= δ
        self.db -= np.reshape(np.sum(δ, axis = 1), (self.Out, 1))
    
    #Forward/Back Propagation with Dropout (This slows down training a lot, but reduces overfitting some)
    """
    #Forward Propagate an Input (Saving the Activation and Layers)
    def forward(self, x):
        #Save the Input
        self.x = np.copy(x)
        
        #Set the Dropout Weights/Biases
        self.wDrop = (np.random.random((self.Out, self.In)) > self.dropout)
        self.bDrop = (np.random.random((self.Out, 1)) > self.dropout)
        
        #Get and Save the Activation
        self.z = np.dot(self.w*self.wDrop, x) + self.b*self.bDrop
        
        #Get and Save the Output Layer
        self.a = self.norm(self.z)
        
        #Return the Output
        return self.a
    
    #Back-Propagate Data
    def backprop(self, δ):
        #Update the Δw Matrix
        self.dw -= np.dot(self.x, δ.T).T*self.wDrop
        
        #Update the Δb Vector
        self.db -= δ*self.bDrop
        #self.db -= np.reshape(np.sum(δ, axis = 1), (self.Out, 1))
    """
    
    #Compute the First δ Vector (Used When a Layer is the Last Layer)
    def delta0(self, y):
        return self.norm.grad(self.a, y, self.z)
    
    #Compute Other δ Vectors (Used When a Layer is a Hidden Layer)
    def delta(self, δ):
        return np.dot(self.w.T, δ)
    
    #Update the δ Vecotr (Used When a Layer is not the First Layer)
    def updateDelta(self, δ):
        return δ*self.norm.derivative(self.z)
    
    #L2 Cost Modifier
    def L2Cost(self):
        return np.sum(self.w**2) + np.sum(self.b**2)
    
    #Update the Layer
    def update(self, n):
        #Update the Weights/Biases (L2 Regularized)
        self.w += self.η*(self.dw - self.λ*self.w/n)
        self.b += self.η*(self.db - self.λ*self.b/n)
        
        #Reset the Update Arrays
        self.dw *= self.α
        self.db *= self.α
    
    #Reset the Layer
    def reset(self):
        #Reset the Update Arrays
        self.x = np.zeros((self.In, 1), dtype = np.float32)
        self.a = np.zeros((self.Out, 1), dtype = np.float32)
        self.z = np.zeros((self.Out, 1), dtype = np.float32)
        self.dw = np.zeros((self.Out, self.In), dtype = np.float32)
        self.db = np.zeros((self.Out, 1), dtype = np.float32)
    
    #Save the Layer as .csv's
    def saveCSV(self, filepath, i):
        #Make a New Folder
        filepath += '\\Layer ' + str(i) + ' - Linear'
        os.makedirs(filepath)
        
        #Save the Parameters
        params = np.hstack((self.In, self.Out, self.η, self.α, Network.norms.index(type(self.norm))))
        np.savetxt(filepath + '//Parameters.csv', params, delimiter = ',')
        
        #Save the Weights/Biases
        np.savetxt(filepath + '//Weights.csv', self.w, delimiter = ',')
        np.savetxt(filepath + '//Biases.csv', self.b, delimiter = ',')
    
    #Load a Layer from .csv's
    def loadCSV(filepath, i):
        #Set the Filepath
        filepath += '\\Layer ' + str(i) + ' - Linear'
        
        #Load the Parameters
        params = np.loadtxt(filepath + '//Parameters.csv', delimiter = ',')
        
        #Create a New Layer Object
        In, Out = int(params[0]), int(params[1])
        layer = Linear(In, Out, Network.norms[int(params[-1])])
        
        #Load the Weights/Biases
        w = np.loadtxt(filepath + '//Weights.csv', dtype = np.float32, delimiter = ',')
        b = np.loadtxt(filepath + '//Biases.csv', dtype = np.float32, delimiter = ',')
        
        #Set the Layer's Weights/Biases
        layer.w = np.reshape(w, (Out, In))
        layer.b = np.reshape(b, (Out, 1))
        
        #Set η and α
        layer.η = params[2]
        layer.α = params[3]
        
        #Return the Layer
        return layer

#Convolutional Layer
class Convolutional:
    #Name of the Layer
    name = 'Convolutional'
    
    #Create the Layer
    def __init__(self, size, In, Out, norm, pad = (0, 0), dilation = (0, 0)):
        #CHECK OTHER INPUTS
        
        #Check the Input Size
        if type(In) not in Network.integers:
            raise TypeError('In must be an integer greater than 0')
        elif In < 1:
            raise ValueError('In must be an integer greater than 0')
        
        #Check the Output Size
        if type(Out) not in Network.integers:
            raise TypeError('Out must be an integer greater than 0')
        elif Out < 1:
            raise ValueError('Out must be an integer greater than 0')
        
        #Check the Normilization Function
        if norm not in Network.norms:
            raise TypeError('Invalid normalization function provided')
        
        #Initialize the Weights/Biases
        
        
        #Set the Normalization Function
        self.norm = norm()
        
        #Save the Input/Output Sizes
        self.shape = size
        self.In = In
        self.Out = Out
        
        #Initialize the Hyperparameters (These Are Set by the Network Object)
        self.α = 0
        self.η = 0
        self.λ = 0
        self.γ = 0
        
        #Initialize Arrays for Backpropagation
        
    
    #Copy the Layer
    def copy(self):
        pass
    
    #Mutate the Layer
    def mutate(self, rate):
        pass
    
    #Compute the Number of Free Parameters in the Layer
    def size(self):
        pass
    
    #Forward Propagate an Input
    def __call__(self, x):
        pass
    
    #Forward Propagate an Input (Saving the Activation and Layers)
    def forward(self, x):
        pass
    
    #Back-Propagate Data
    def backprop(self, δ):
        pass
    
    #Compute the First Delta Vector (Used when a Layer is the Last Layer)
    def delta0(self, y):
        pass
    
    #Compute Other δ Vectors (Used When a Layer is a Hidden Layer)
    def delta(self, δ):
        pass
    
    #Update the δ Vecotr (Used When a Layer is not the First Layer)
    def updateDelta(self, δ):
        pass
    
    #L2 Cost Modifier
    def L2Cost(self):
        pass
    
    #Update the Layer
    def update(self, n):
        pass
    
    #Reset the Layer
    def reset(self):
        pass
    
    #Save the Layer as .csv's
    def saveCSV(self, filepath, i):
        #Make a New Folder
        filepath += '\\Layer ' + str(i) + ' - Convolutional'
        os.makedirs(filepath)
    
    #Load a Layer from .csv's
    def loadCSV(filepath, i):
        pass

#Change the Shape of a Layer (Passed Methods are Need to Exist as they will be Called by the Network)
class Reshape:
    #Name of the Layer
    name = 'Reshape'
    
    #Create the Layer
    def __init__(self, shape):
        #Set the Attributes
        self.input = None
        self.output = shape
        self.a = np.zeros(shape, dtype = np.float32)
        
        #Initialize Other Attributes (Which Will be Referenced by the Network)
        self.α = 0
        self.η = 0
        self.λ = 0
        self.γ = 0
    
    #Copy the Layer
    def copy(self):
        #Create a New Object
        new = Reshape(self.output)
        
        #Copy the Activations
        if type(self.a) != type(None):
            new.a = np.copy(self.a)
        
        #Copy the Input Size
        if type(self.input) != type(None):
            new.input = tuple(np.copy(self.input))
        
        #Return the New Object
        return new
    
    #A Norm (For Computing the Cost if Reshape is Used as a Final Layer)
    class norm:
        #The Quadratic Cost Function
        def cost(a, y):
            return 0.5*np.sum((a - y)**2)/np.size(y)
    
    #Mutate the Layer
    def mutate(self, rate):
        pass
    
    #Compute the Number of Free Parameters in the Layer
    def size(self):
        return 0
    
    #Forward Propagate an Input
    def __call__(self, x):
        #Return the Reshaped Input
        return np.reshape(x, self.output)
    
    #Forward Propagate an Input (Saving the Activation and Layers)
    def forward(self, x):
        #Save the Shape of the Input
        self.input = np.shape(x)
        
        #Save the Output
        self.a = np.reshape(x, self.output)
        
        #Return the Reshaped Input
        return np.copy(self.a)
    
    #Back-Propagate Data
    def backprop(self, δ):
        pass
    
    #Compute the First Delta Vector (Used when a Layer is the Last Layer)
    def delta0(self, y):
        return self.a - y
    
    #Compute Other δ Vectors (Used When a Layer is a Hidden Layer)
    def delta(self, δ):
        return np.reshape(δ, self.input)
    
    #Update the δ Vecotr (Used When a Layer is not the First Layer)
    def updateDelta(self, δ):
        return δ
    
    #L2 Cost Modifier
    def L2Cost(self):
        return 0
    
    #Update the Layer
    def update(self, n):
        pass
    
    #Reset the Layer
    def reset(self):
        #Reset the Attributes
        self.input = None
        self.a = np.zeros(self.output, dtype = np.float32)
    
    #Save the Layer as .csv's
    def saveCSV(self, filepath, i):
        #Make a New Folder
        filepath += '\\Layer ' + str(i) + ' - Reshape'
        os.makedirs(filepath)
        
        #Save the Output Shape
        np.savetxt(filepath + '\\Output.csv', self.output, delimiter = ',')
    
    #Load a Layer from .csv's
    def loadCSV(filepath, i):
        #Load the Output Shape
        filepath += '\\Layer ' + str(i) + ' - Reshape'
        output = tuple(np.loadtxt(filepath + '\\Output.csv', dtype = int, delimiter = ','))
        
        #Make and Return a New Layer Object
        return Reshape(output)

#Max Pooling Layer
class MaxPool:
    #Name of the Layer
    name = 'MaxPool'
    
    #Create the Layer
    def __init__(self, size = (2, 2), stride = (1, 1)):
        #Check the Inputs
        
        
        #Set the Object Attributes
        self.shape = size
        self.stride = stride
        self.input = None
        self.a = None
        
        #Initialize Other Attributes (Which Will be Referenced by the Network)
        self.α = 0
        self.η = 0
        self.λ = 0
        self.γ = 0
    
    #Call the Object
    def __call__(self, x):
        #Get the Shape of the Input
        X, Y = np.shape(x)
        
        #Split the Stride
        dx = self.stride[0]
        dy = self.stride[1]
        
        #Compute the Size of the Output Matrix
        N = int(np.ceil((X - self.shape[0])/dx) + 1)
        M = int(np.ceil((Y - self.shape[1])/dy) + 1)
        
        #Create the Output Matrix
        y = np.array((N, M))
        for i in range(N):
            for j in range(M):
                y[i][j] = np.max(x[i*dx:min(i*dx + self.shape[0], X), j*dy:min(j*dy + self.shape[1], Y)])
        
        #Return the Output Matrix
        return y
    
    #Copy the Layer
    def copy(self):
        pass
    
    #A Norm (For Computing the Cost if Reshape is Used as a Final Layer)
    class norm:
        #The Quadratic Cost Function
        def cost(a, y):
            return 0.5*np.sum((a - y)**2)/np.size(y)
    
    #Mutate the Layer
    def mutate(self, rate):
        pass
    
    #Compute the Number of Free Parameters in the Layer
    def size(self):
        return 0
    
    #Forward Propagate an Input (Saving the Activation and Layers)
    def forward(self, x):
        #Save the Shape of the Input
        self.input = np.shape(x)
        
        #Get and Save the Output
        self.a = self(x)
        
        #Return the Output
        return np.copy(self.a)
    
    #Back-Propagate Data
    def backprop(self, δ):
        pass
    
    #Compute the First Delta Vector (Used when a Layer is the Last Layer)
    def delta0(self, y):
        return self.a - y
    
    #Compute Other δ Vectors (Used When a Layer is a Hidden Layer)
    def delta(self, δ):
        pass 
    
    #Update the δ Vecotr (Used When a Layer is not the First Layer)
    def updateDelta(self, δ):
        pass
    
    #L2 Cost Modifier
    def L2Cost(self):
        return 0
    
    #Update the Layer
    def update(self, n):
        pass
    
    #Reset the Layer
    def reset(self):
        pass
    
    #Save the Layer as .csv's
    def saveCSV(self, filepath, i):
        pass
    
    #Load a Layer from .csv's
    def loadCSV(filepath, i):
        pass

"""A Class for the Network"""
class Network:
    #Type Pools (Including All Normalization Functions and Layer Types)
    integers = [int, np.int, np.int16, np.int32, np.int64]
    floats = [float, np.float16, np.float32, np.float64]
    numbers = list(np.append(integers, floats))
    norms = [sigmoid, tanh, relu, leakyRelu, softmax, linear, perceptron]
    layerTypes = [Linear, Convolutional, Reshape, MaxPool]
    
    #Create a New Network (Input Checking is Done in Network.__setattr__)
    def __init__(self, name, η=0.003, α=0.9, λ=0.5, γ=0.85, dropout=0.5, delay=10, verbose=False):
        #Variables to Record the Training
        self.verbose = verbose
        self.delay = delay
        self.t = time.time() - delay
        self.n = -1
        self.N = -1
        
        #Set the Main Attributes
        self.name = str(name)
        self.layers = []
        self.η = η
        self.α = α
        self.λ = λ
        self.γ = γ
        
        #Initialize the Training Log
        self.epochs = 0
        self.costs = []
        self.examples = [0]
    
    #Copy a Network
    def copy(self, name = None):
        #Get the Name
        if type(name) == type(None):
            name = self.name
        
        #Create a New Network
        new = Network(self.name, self.η, self.α, self.λ, self.γ, self.dropout, self.delay, self.verbose)
        
        #Copy the Layers
        for layer in self.layers:
            new += layer.copy()
        
        #Copy the Training Log
        new.costs = list(np.copy(self.costs))
        new.examples = list(np.copy(self.examples))
        new.t = self.t
        
        #Return the New Network
        return new
    
    #Mutate a Network
    def mutate(self, rate = 0.01):
        #Check the Rate
        if type(rate) not in Network.numbers:
            raise TypeError('rate must be a number')
        elif rate <= 0:
            raise ValueError('rate must be greater than 0')
        
        #Mutate Each Layer
        for layer in self.layers:
            layer.mutate(rate)
    
    #Add a Layer
    def addLayer(self, layer):
        if type(layer) in Network.layerTypes:
            #Edit the Layer's Hyperparameters
            layer.η = self.η
            layer.α = self.α
            layer.λ = self.λ
            layer.γ = self.γ
            
            #Add the Layer
            self.layers.append(layer)
        else:
            #Raise a Type Error
            raise TypeError("layer must be a valid layer type:\n" + str(Network.layerTypes))
    
    #Add a Layer (using +=)
    def __iadd__(self, layer):
        #Add the Layer
        self.addLayer(layer)
        
        #Return the Network
        return self
    
    #Add a Layer to a Network (Saving it to a New Network - Using +)
    def __add__(self, layer):
        #Copy the Network
        new = self.copy()
        
        #Add the Layer to the New Network
        new.addLayer(layer)
        
        #Return the New Network
        return new
    
    #Get the Number of Layers
    def __len__(self):
        return len(self.layers)
    
    #Get the Size of the Network
    def size(self):
        #Sum the Number of Parameters
        x = 0
        for layer in self.layers:
            x += layer.size()
        
        #Return the Number of Parameters
        return x
    
    #Maure Sure η and α Are Set Appropriately
    def __setattr__(self, name, value):
        #Make Sure all the Layers Get Set Too
        if name in {'η', 'α', 'λ', 'γ', 'dropout'}:
            #Check the Input
            if name == 'η':
                #Check η
                if type(value) not in Network.numbers:
                    raise TypeError('η must be numeric')
                elif value <= 0:
                    raise ValueError('η must be greater than 0')
                
                #Print a Warning
                if self.verbose and value > 1:
                    print('Warning: Setting η > 1 may result in unstable traing')
            elif name == 'α':
                #Check α
                if type(value) not in Network.numbers:
                    raise TypeError('α must be numeric')
                elif value < 0 or value > 1:
                    raise ValueError('α must be in the interval [0, 1]')
            elif name == 'λ':
                #Check λ
                if type(value) not in Network.numbers:
                    raise TypeError('λ must be numeric')
                elif value < 0:
                    raise ValueError('λ must be greater than or equal to 0')
            elif name == 'γ':
                #Check γ
                if type(value) not in Network.numbers:
                    raise TypeError('γ must be numeric')
                elif value <= 0:
                    raise ValueError('γ must be greater than 0')
                
                #Print a Warning
                if self.verbose and value > 1:
                    print('Warning: Setting γ > 1 may result in unstable training')
            elif name == 'dropout':
                #Check the Dropout Rate
                if type(value) not in Network.numbers:
                    raise TypeError('dropout must be numeric')
                elif value < 0 or value >= 1:
                    raise ValueError('dropout must be in the interval [0, 1)')
                
                #Print a Warning
                if self.verbose and value > 0.9 and False:
                    print('Warning: dropout may be too high')
            
            #Set the Attribute in Each Layer
            for layer in self.layers:
                layer.__dict__[name] = np.float32(value)
            
            #Set the Attribute
            self.__dict__[name] = np.float32(value)
        elif name == 'epochs':
            #Check the Epochs
            if type(value) not in Network.integers:
                raise TypeError('epochs must be an integer')
            elif value < 0:
                raise ValueError('epochs must be greater than or equal to 0')
            
            #Update the Learning Rates of Each Layer
            for layer in self.layers:
                layer.η = self.η*self.γ**value
            
            #Set the Attribute
            self.__dict__[name] = value
        else:
            #Check the Input
            if name == 'delay':
                if type(value) not in Network.numbers:
                    raise TypeError('delay must be numeric')
                elif value <= 0:
                    raise ValueError('delay must be greater than 0')
            elif name == 'verbose' and type(value) != bool:
                raise TypeError('verbose must be a boolean')
        
            #Set the Attribute
            self.__dict__[name] = value
    
    #Get a Specific Layer
    def __getitem__(self, key):
        return self.layers[key]
    
    #Forward Propagate an Input
    def __call__(self, x):
        #Forward Propagate the Input
        for layer in self.layers:
            x = layer(x)
        
        #Return the Output
        return x
    
    #Forward Propagate an Input (Saving the Hidden Layers)
    def forward(self, x):
        #Forward Propagate the Input
        for layer in self.layers:
            x = layer.forward(x)
        
        #Return the Output
        return x
    
    #Back-Propagate an Output
    def backprop(self, y):
        #Compute the Initial Error
        δ = self[-1].delta0(y)
        
        #Backpropagate the Error
        for i in range(1, len(self) + 1):
            #Backpropagate the Current Layer
            self[-i].backprop(δ)
            
            #Update δ
            if i < len(self):
                δ = self[-i - 1].updateDelta(self[-i].delta(δ))
        
        #Increment the Examples Trained Counter
        self.examples[-1] += 1
    
    #Cause the Weights/Biases to Update
    def update(self):
        #Update Each Layer
        for layer in self.layers:
            layer.update(self.examples[-1] - self.examples[-2])
    
    #Reset the Network
    def reset(self, reset_η = False):
        #Reset Each Layer (And Maybe Their Learning Rates)
        for layer in self.layers:
            layer.reset()
            if reset_η == True:
                layer.η = self.η
    
    """Train Networks"""
    #Check the Input Data and Zip It Into a List of Paired Data
    def makeData(self, x, y):
        #Make Sure x is Valid
        if type(x) not in {list, np.ndarray}:
            raise TypeError('x must be a list or an array')
        elif np.size(x) == 0:
            raise ValueError('No x data provided')
        elif type(x[0]) != np.ndarray:
            raise TypeError('x must contain numpy arrays')
        
        #Make Sure y is Valid
        if type(y) not in {list, np.ndarray}:
            raise TypeError('y must be a list or an array')
        elif np.size(y) == 0:
            raise ValueError('No y data provided')
        elif type(y[0]) != np.ndarray:
            raise TypeError('y must contain numpy arrays')
        
        #Make Sure x and y have the Same Length
        if len(x) != len(y):
            raise Exception('x and y must have the same length')
        
        #Make Sure self(x) Works
        try:
            #Test an Input
            self(x[0])
        except:
            #Find the Offending Layer
            z = x[0]
            for i in range(len(self)):
                try:
                    z = self[i](z)
                except:
                    message = 'Unable to pass input through layer ' + str(i + 1) + ': Input Shape: '
                    raise Exception(message + str(np.shape(z)))
        
        #Make Sure self(x) and y Have the Same Shape
        try:
            #Get the Shape of the Output of the Network
            a = np.shape(self(x[0]))
            
            #Make Sure the Output of the Network has the Right Shape
            if np.shape(y[0]) != a:
                raise Exception('Shape Mismatch: Network output has a different shape than y')
        except:
            raise Exception('Unable to pass inputs through the model, check input shape')
        
        #Create and Return a List of Data
        return list(zip(x, y))
    
    #Compute the Average Loss Per Output Neuron of the Network
    def cost(self, testData):
        #Compute the Average Loss Per Output Neuron
        cost = 0
        for data in testData:
            cost += self[-1].norm.cost(self(data[0]), data[1])
        
        #Regularize the Cost
        #cost += self[-1].L2Cost()
        
        #Return the Cost
        return cost/len(testData)
    
    #Test a Network
    def test(self, testData):
        #Compute the Cost and Edit the Training Log
        self.costs.append(self.cost(testData))
        self.examples.append(self.examples[-1])
    
    #Callback Function
    def callBack(self, testData):
        #Record the Current Time
        self.t = time.time()
        
        #Test the Network
        if len(self.examples) == 1 or self.examples[-1] != self.examples[-2]:
            self.test(testData)
        
        #Compute the Completion
        completion = round(100*(self.examples[-1] - self.n)/self.N, 1)
        
        #Print the Cost
        print(str(completion) + '% Complete: Cost = ' + str(sfRound(self.costs[-1], 2)))
    
    #Full Gradient Descent
    def GD(self, data, testData):
        #Go Over Each Piece of Training Data
        for d in data:
            #Print a Status Update
            if time.time() - self.t > self.delay:
                self.callBack(testData)
            
            #Forward-Propagate the Data
            self.forward(d[0])
            
            #Back-Propagate the Data
            self.backprop(d[1])
        
        #Update the Network
        self.update()
    
    #Stochastic Gradient Descent
    def SGD(self, data, testData):
        #Train a Random Epoch
        for i in range(len(data)):
            #Print a Status Update
            if time.time() - self.t > self.delay:
                self.callBack(testData)
            
            #Choose a Piece of Data at Random
            d = data[np.random.randint(0, len(data))]
            
            #Forward-Propagate the Data
            self.forward(d[0])
            
            #Back-Propagate the Data
            self.backprop(d[1])
            
            #Update the Network
            self.update()
    
    #Mini-Batch Gradient Descent
    def MBGD(self, data, testData, batchsize):
        #Train the Data in Batches
        for i in range(int(np.ceil(len(data)/batchsize))):
            #Train the Mini-Batch Using Gradient Descent
            self.GD(data[i*batchsize:min((i + 1)*batchsize, len(data))], testData)
    
    #A List of Training Methods
    methods = [GD, SGD, MBGD]
    
    #Train a network
    def train(self, x, y, tx, ty, epochs = 1, batchsize = 32):
        #Print a Status Update
        if self.verbose:
            print('Zipping Training Data:')
        
        #Check the Training Data and Zip the Inputs Into a List of Data
        data = self.makeData(x, y)
        
        #Print a Status Update
        if self.verbose:
            print('Zipping Test Data:')
        
        #Check the Test Data and Zip the Inputs Into a List of Data
        testData = self.makeData(tx, ty)
        
        #Check the Batchsize
        if type(batchsize) == str and batchsize.lower() in {'auto', 'all'}:
            #Set the Method Key
            methodKey = 2*int(batchsize.lower() == 'auto')
            
            #Set the Batchsize
            if methodKey == 0:
                batchsize = len(data)
            else:
                batchsize = max(len(data) // 25, 5)
        elif type(batchsize) not in Network.integers or batchsize < 0:
            #Raise a TypeError
            raise ValueError("batchsize must either be 'auto,' 'all,' or a non-negative integer.")
        else:
            #Set the Method Key
            if batchsize == 0:
                methodKey = 1
            elif batchsize < len(data):
                methodKey = 2
            else:
                methodKey = 0
        
        #Check the Epochs
        if type(epochs) not in Network.integers:
            #Raise a Type Error
            raise TypeError('epochs must be an integer')
        elif epochs < 1:
            #Raise a Value Error
            raise ValueError('epochs must be greater than 0')
        
        #Set the Parameters to Give the Training Method
        if methodKey == 2:
            params = (data, testData, batchsize)
        else:
            params = (data, testData)
        
        #Set the Callback Constant
        self.N = len(data)
        
        #Test the Network
        if len(self.examples) == 1 or self.examples[-1] != self.examples[-2]:
            self.test(testData)
        
        #Train the Network
        for epoch in range(epochs):
            #Set the Callback Number
            self.n = self.examples[-1]
            
            #Shuffle the Data (If Necessary)
            if methodKey == 2:
                np.random.shuffle(params[0])
            
            #Print a Status Update
            newLine = '\n'*(epoch > 0 or self.verbose)
            print(newLine + 'Epoch ' + str(epoch + 1) + ' of ' + str(epochs) + ':')
            
            #Train the Network
            Network.methods[methodKey](self, *params)
            
            #Increment the Number of Epochs Trained
            self.epochs += 1
        
        #Test the Network
        if self.examples[-1] != self.examples[-2]:
            self.test(testData)
        
        #Reset the Network
        self.reset()
    
    """Plot the Training Log"""
    def plot(self, save = False):
        #Create the Plot
        fig = plt.figure(figsize = (6, 4))
        plt.plot(self.examples[:-1], self.costs, label = self.name + ' Cost')
        plt.xlabel('Examples Trained')
        plt.ylabel('Average Loss per Output Neuron')
        plt.legend()
        plt.show()
        
        #Save the Plot
        if save == True:
            fig.savefig(self.name + ' cost.png')
    
    """Save/Load Networks"""
    #Format a File Name
    def formatFileName(name):
        #Make Sure the Name is a String
        if type(name) != str:
            raise TypeError('name must be a string.')
        
        #Keep the File Name Up to the First '.'
        new = ''
        for char in name:
            if char == '.':
                break
            else:
                new += str(char)
        
        #Make Sure the String Isn't Empty
        if len(new) == 0:
            raise ValueError('The name "' + name + '" was not formatted filename.p')
        
        #Return the File Name
        return new + '.p'
    
    #Save a Network
    def save(self):
        pickle.dump(self, open(Network.formatFileName(self.name), 'wb'))
    
    #Load a Network
    def load(name):
        #Format the File Name
        name = Network.formatFileName(name)
        
        #Make Sure the File Exists
        if name in os.listdir():
            return pickle.load(open(name, 'rb'))
        else:
            raise Exception(name + ' not found in ' + os.getcwd())
    
    #Recursively Clear a Folder and then Remove the Folder
    def removeFolder(filepath):
        for item in os.listdir(filepath):
            if '.' in item:
                #Remove a File
                os.remove(filepath + '\\' + item)
            else:
                #Clear a Folder
                Network.removeFolder(filepath + '\\' + item)
                
        #Remove the Now Empty Folder
        os.removedirs(filepath)
    
    #Save a Network in .csv's
    def saveCSV(self):
        #Make or Clear a Filepath
        if self.name in os.listdir():
            #Clear the Filepath
            Network.removeFolder(os.getcwd() + '\\' + self.name)
        
        #Make the Filepath
        filepath = os.getcwd() + '\\' + self.name
        os.makedirs(os.getcwd() + '\\' + self.name)
        filepath += '\\'
        
        #Save the Network Parameters
        params = np.array([self.η, self.α, self.λ, self.delay, self.examples[-1], int(self.verbose)])
        np.savetxt(filepath + 'Parameters.csv', params, delimiter = ',')
        
        #Save the Training Log
        log = np.array([self.examples[:-1], self.costs])
        if np.shape(log)[1] > 0:
            np.savetxt(filepath + 'Training Log.csv', log, delimiter = ',')
        
        #Save Each Layer (Recording a Key for its Type)
        keys = []
        for i in range(len(self)):
            #Save the Layer
            self[i].saveCSV(filepath, i)
            
            #Record the Key
            keys.append(Network.layerTypes.index(type(self[i])))
        
        #Save the Layer Keys
        np.savetxt(filepath + 'Keys.csv', keys, delimiter = ',')
    
    #Load a Network from .csv's
    def loadCSV(name):
        #Format the Filename
        name = Network.formatFileName(name)[:-2]
        
        #Make Sure the Path Exists
        if name not in os.listdir():
            raise Exception(name + ' not found in ' + os.getcwd())
        
        #Get the Filepath
        filepath = os.getcwd() + '\\' + name
        
        #Load the Parameters
        params = list(np.loadtxt(filepath + '\\Parameters.csv', delimiter = ','))
        params[-1] = bool(params[-1])
        
        #Make a New Network Object
        network = Network(name, *params)
        
        #Load the Training Log
        if 'Training Log.csv' in os.listdir(os.getcwd() + '\\' + name):
            #Load the Log
            log = np.loadtxt(filepath + '\\Training Log.csv', delimiter = ',')
            
            #Add the Log to the Network
            network.examples = list(log[0])
            network.costs = list(log[1])
        network.examples.append(params[3])
        
        #Load the Layer Keys
        keys = np.loadtxt(filepath + '\\Keys.csv', dtype = int, delimiter = ',')
        
        #Load the Layers
        for i in range(len(keys)):
            network += Network.layerTypes[keys[i]].loadCSV(filepath, i)
        
        #Return the Network
        return network

"""Functions to Help with Documentation"""
#List All the Normalization Functions
def listNorms():
    print('Supported Normalization Functions:')
    for function in Network.norms:
        print(' - ' + function.name)

#List All the Layer Types
def listLayers():
    print('Supported Layer Types:')
    for layer in Network.layerTypes:
        print(' - ' + layer.name)