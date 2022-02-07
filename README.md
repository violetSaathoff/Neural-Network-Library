# Neural-Network-Library
My Custom Neural Network Library

I wrote this library so that I could develop a deeper understanding of how Neural Networks work. While it does function, I don't recommend actually using this library since it isn't as well optimized as more professional packages like PyTorch, and TensorFlow, and it also lacks much of the functionality those packages have.

The basic structure of my networks is object-oriented, where the network itself is an object, each layer is an object, and each normalization function is an object. I chose this structure because it felt like the most natural way for me to write this at the time, and it was similar to other ML packages I'd used before.

To create a new network you first need to create an instance of the Network class. Then you can add layers either using the Network.addLayer() method, or by using the += operator (they perform the same task, I just personally prefer using the += operator because I like how it looks). Currently supported layer types are: Linear, Reshape, Maxpool, and Meanpool (Convolutional layers will be added the next time I work on this project). 

Here is an example feed-forward network which would be suitable for classifying the MNIST digits:

    # Example MNIST Network
    network = Network('MNIST ANN')
    network += Linear(784, 100, sigmoid)
    network += Linear(100, 100, sigmoid)
    network += Linear(100, 100, sigmoid)
    network += Linear(100, 10, softmax)

This network has an input layer which recieves the flattened MNIST images, two 100-neuron hidden layers, and a 10-neuron output layer. While the activation function used by most of the layers is the sigmoid activation function, the final layer uses a softmax activation function since it is attempting to represent the probability that the input image is of a specific numeral.

Training networks is done with the Network.train() method, which requires the training data, test data, and a few other parameters. 

There are 3 main training algorithms: 
- Full-Batch Gradient Descent, where the entire data set is trained on between weights/biases updates.
- Mini-Batch Gradient Descent, where the data is shuffled, split into batches, and then the weights/biases are updated after each batch is trained on.
- Stochastic Gradient Descent, where each training example is chosen at random from the data set, and then the weights/biases are updated. This differes from using MBGD with a       batchsize of 1 since the shuffling ensures that all data in MBGD will be trained on once per epoch, while SGD will train on some examples more than once an epoch, and           miss other examples entirely.

I recommend MBGD since in practice it has provided faster/more reliable convergance than the other methods.
