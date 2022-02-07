# Neural-Network-Library
Custom Neural Network Library

I wrote this library so that I could develop a deeper understanding of how Neural Networks work. While it does function, I don't recommend actually using this library since it isn't as well optimized as more professional packages like PyTorch, and TensorFlow, and it also lacks much of the functionality those packages have.

The basic structure of my networks is object-oriented, where the network itself is an object, each layer is an object, and each normalization function is an object. I chose this structure because it felt like the most natural way for me to write this at the time, and it was similar to other ML packages I'd used before.

To create a new network you first need to create an instance of the Network class. Then you can add layers either using the Network.addLayer() method, or by using the += operator (they perform the same task, I just personally prefer using the += operator because I like how it looks). Currently supported layer types are: Linear, Reshape, Maxpool, and Meanpool (Convolutional layers will be added the next time I work on this project). 

Here is an example feed-forward network which would be suitable for classifying the MNIST digits:

network = Network(
