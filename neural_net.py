import numpy as np

### Class to represent a single neuron
class Neuron:
    def __init__(self, num_inputs):
        # Initialize weights and bias to small random numbers
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn(1)[0]

    def feedforward(self, inputs):
        # Save inputs for use in backpropagation
        self.previous_inputs = inputs
        # Weight inputs, add bias, then use the activation function
        self.last_total = np.dot(inputs, self.weights) + self.bias
        self.last_output = sigmoid(self.last_total)
        return self.last_output
    
    def backpropagate(self, error, learning_rate):
        # Apply the chain rule to calculate derivative of the loss with respect to weights and bias
        dtotal = error * sigmoid_derivative(self.last_total)
        
        # Compute the gradients
        dweights = dtotal * np.array(self.previous_inputs)
        dbias = dtotal

        # Update weights and bias
        self.weights -= learning_rate * dweights
        self.bias -= learning_rate * dbias

        # Return error to pass to the previous layer
        return np.sum(dtotal * self.weights)
    

### Class to represent a fully connected neural network layer
class FullyConnectedLayer:
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for i in range(num_neurons)]
        
    def feedforward(self, inputs):
        # Feed inputs through all neurons in this layer
        return np.array([neuron.feedforward(inputs) for neuron in self.neurons])
    
    def backpropagate(self, errors, learning_rate):
        return np.array([neuron.backpropagate(error, learning_rate) for neuron, error in zip(self.neurons, errors)])


### Class to represent a neural network
class Network:
    def __init__(self, num_inputs):
        self.layers = []
        self.num_inputs = num_inputs

    def add_layer(self, num_neurons, layer_type=FullyConnectedLayer):
        num_inputs = self.num_inputs if len(self.layers) == 0 else len(self.layers[-1].neurons)
        self.layers.append(layer_type(num_neurons, num_inputs))
    
    def train(self, inputs, expected_outputs, epochs=10, learning_rate=0.01):
        print("Training...")
        for epoch in range(epochs):
            print("Epoch", epoch + 1)
            for single_input, single_output in zip(inputs, expected_outputs):
                outputs = self.feedforward(single_input)
                errors = np.array(outputs) - np.array(single_output)
                for layer in reversed(self.layers):
                    errors = layer.backpropagate(errors, learning_rate)

    def feedforward(self, inputs):
        for layer in self.layers:
            inputs = layer.feedforward(inputs)
        return inputs

    def predict(self, inputs):
    # Feed inputs through all layers in the network
        predictions = []
        for input_data in inputs:
            prediction = input_data
            for layer in self.layers:
                prediction = layer.feedforward(prediction)
            predictions.append(prediction)
        return np.array(predictions)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

### Derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))