import numpy as np
import random


### Step activation function
def step(x):
    return 1 if x > 0 else 0

### Class to represent a single neuron
class Neuron:
    def __init__(self, num_inputs):
        # Initialize weights and bias to small random numbers
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn(1)[0]

    def feedforward(self, inputs):
        # Weight inputs, add bias, then use the activation function
        total = 0
        for input, weight in zip(inputs, self.weights):
            total += input * weight
        return self.activation(total + self.bias)
    
    def activation(self, x, activation_function=step):
        # Call the activation function
        return activation_function(x)
    
    def train(self, inputs, output, learning_rate=1):
        # Adjust weights and bias using the Perceptron Learning Rule
        prediction = self.feedforward(inputs)
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * (output - prediction) * inputs[i]
        self.bias += learning_rate * (output - prediction)

    

### Class to represent a neural network layer
class Layer:
    def __init__(self, num_neurons=1, output_layer=False):
        self.neurons = []
        self.output_layer = output_layer
        for i in range(num_neurons):            
            self.neurons.append(Neuron(num_inputs=2))            

    def feedforward(self, inputs):
        # Feed inputs through all neurons in this layer
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.feedforward(inputs))
        if not self.output_layer:
            return outputs
        
        return outputs[0]
    
    def train(self, inputs, outputs, learning_rate=1):
        for input, output in zip(inputs, outputs):
            # Train all neurons in this layer
            for neuron in self.neurons:
                neuron.train(input, output, learning_rate)




### Class to represent a neural network
class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        # Add a layer to the network
        self.layers.append(layer)
    
    def train(self, inputs, outputs, epochs=10, learning_rate=1):
        for epoch in range(epochs):
            # Train all layers in the network
            for layer in self.layers:
                layer.train(inputs, outputs, learning_rate)

    def predict(self, inputs):
        # Feed inputs through all layers in the network
        for layer in self.layers:
            outputs = layer.feedforward(inputs)
            inputs = outputs
        return outputs
    
### Function to create random OR dataset
def create_or_dataset(num_samples=100):
    inputs = []
    outputs = []
    for i in range(num_samples):
        input = [random.randint(0, 1), random.randint(0, 1)]
        output = int(input[0] or input[1])
        inputs.append(input)
        outputs.append(output)
    return inputs, outputs

if __name__ == "__main__":
    neuralnet = Network()
    neuralnet.add_layer(Layer(num_neurons=1, output_layer=True))

    train_inputs, train_outputs = create_or_dataset(num_samples=5)
    test_inputs, test_outputs = create_or_dataset(num_samples=100)

    #Train the network
    neuralnet.train(train_inputs, train_outputs, epochs=10, learning_rate=1)

    #Test the network
    num_correct = 0
    for input, output in zip(test_inputs, test_outputs):
        prediction = neuralnet.predict(input)
        print("Expected:", output, "Predicted:", 1 if (prediction >= 0.5) else 0)
        if (prediction >= 0.5) == output:
            num_correct += 1
    print("Accuracy: {}%".format(num_correct / len(test_inputs) * 100))
