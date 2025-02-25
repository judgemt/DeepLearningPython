import numpy as np

def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z)) # applies element-wise

class NeuralNetwork:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) 
                       for y in sizes[1:]] # for each layer in sizes, initialize biases with random numbers
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])] # then, for the in-between-layers weights, initialize with random numbers
        # So biases is a list of bias vectors (one for each neuron)
        # and weights is the matrix of size output layer, input layer size

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights): # for each layer (which has biases and weights):
            a = sigmoid(np.dot(w, a) + b) # multiply each matrix against its activations and add the biases
            # note: a gets carried over into the next iteration (layer) and changes sizes
        return a # list of activations 
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
         if test_data: n_test = len(test_data) # test data is a list
         n = len(training_data)

         for j in range(epochs):
              np.random.shuffle(training_data)

            # Each minibatch is a sampling of the training data
              mini_batches = [
                   training_data[k:k+mini_batch_size]
                   for k in range(0, n, mini_batch_size)
              ]

            # Update minibatches
              for mini_batch in mini_batches:
                   self.update_mini_batch(mini_batch, eta)
            
            # Report if using test data
              if test_data:
                   print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)) 
              else: 
                   print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
         nabla_b = [np.zeros(b.shape) for b in self.biases]
         nabla_w = [np.zeros(w.shape) for w in self.weights]

         for x, y in mini_batch:
              delta_nabla_b, delta_nabla_w = self.backprop(x, y)
              nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
              nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
