# Neural-Network-Library
Custom Neural Network Library

I wrote this library so that I could develop a deeper understanding of how Neural Networks work. While it does function, I don't recommend actually using this library since it isn't as well optimized as more professional packages like PyTorch, and TensorFlow, and it also lacks much of the functionality those packages have.

The basic structure of my networks is object-oriented, where the network itself is an object, each layer is an object, and each normalization function is an object. I chose this structure because it felt like the most natural way for me to write this at the time, and it was similar to other ML packages I'd used before.

To create a new network you first need to create an instance of the Network class. Then you can add layers either using the Network.addLayer() method, or by using the += operator (they perform the same task, I just personally prefer using the += operator because I like how it looks). Currently supported layer types are: Linear, Reshape, Maxpool, and Meanpool (Convolutional layers will be added the next time I work on this project). 

Here is an example feed-forward network which would be suitable for classifying the MNIST digits:

#Example MNIST Network
network = Network('MNIST ANN')
network += Linear(784, 100, sigmoid)
network += Linear(100, 100, sigmoid)
network += Linear(100, 100, sigmoid)
network += Linear(100, 10, softmax)

This network has an input layer which recieves the flattened MNIST images, two 100-neuron hidden layers, and a 10-neuron output layer. While the activation function used by most of the layers is the sigmoid activation function, the final layer uses a softmax activation function since it is attempting to represent the probability that the input image is of a specific numeral.
