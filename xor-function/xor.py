import numpy as np
import random


### Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)


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



    

### Class to represent a neural network layer
class Layer:
    def __init__(self, num_neurons, num_inputs, output_layer=False):
        self.neurons = [Neuron(num_inputs) for i in range(num_neurons)]
        self.output_layer = output_layer
        
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

    def add_layer(self, num_neurons, output_layer=False):
        num_inputs = self.num_inputs if len(self.layers) == 0 else len(self.layers[-1].neurons)
        self.layers.append(Layer(num_neurons, num_inputs, output_layer=output_layer))
    
    def train(self, inputs, expected_outputs, epochs=10, learning_rate=0.01):
        for epoch in range(epochs):
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
        for layer in self.layers:
            inputs = layer.feedforward(inputs)
        return inputs
    
### Function to create random XOR dataset
def create_xor_dataset(num_samples=100):
    inputs = []
    outputs = []
    for i in range(num_samples):
        x = random.randint(0, 1)
        y = random.randint(0, 1)
        inputs.append([x, y])
        outputs.append(x ^ y)
    return inputs, outputs

if __name__ == "__main__":
    neuralnet = Network(num_inputs=2)
    neuralnet.add_layer(10)
    neuralnet.add_layer(1)

    train_inputs, train_outputs = create_xor_dataset(num_samples=1000)
    test_inputs, test_outputs = create_xor_dataset(num_samples=1000)

    #Train the network
    neuralnet.train(train_inputs, train_outputs, epochs=10, learning_rate=1)

    #Test the network
    num_correct = 0
    for input, output in zip(test_inputs, test_outputs):
        prediction = neuralnet.predict(input)
        predicted_output = 1 if prediction[0] >= 0.5 else 0  # Use the first element of prediction
        print("Expected:", output, "Predicted:", predicted_output)
        if predicted_output == output:
            num_correct += 1
    print("Accuracy: {}%".format(num_correct / len(test_inputs) * 100))

