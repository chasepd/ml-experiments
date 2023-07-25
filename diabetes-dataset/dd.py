import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import random

seed_value = 6162
random.seed(seed_value)
np.random.seed(seed_value)

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


### Load the Diabetes dataset and create training and testing sets
def load_data():
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    ### Normalize the data
    X = X / np.amax(X, axis=0)

    ### One-hot encode the targets
    y = np.array([[1 if y[i] == j else 0 for j in range(3)] for i in range(len(y))])

    return X, y

def one_hot_encode_predictions(raw_predictions):
    # Find the index of the highest probability
    indices = np.argmax(raw_predictions, axis=1)
    
    # Initialize an empty array for the one-hot encoded predictions
    one_hot_predictions = np.zeros(raw_predictions.shape)

    # Set the index with the highest probability to 1
    for i in range(len(indices)):
        one_hot_predictions[i, indices[i]] = 1

    return one_hot_predictions

def calculate_accuracy(predictions, outputs):
    # Convert one-hot encoded predictions and outputs back to class labels
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(outputs, axis=1)
    
    # Calculate and return the accuracy
    accuracy = np.sum(predicted_labels == true_labels) / len(true_labels)
    return accuracy

### Evaluate the model
def evaluate_model(model, inputs, outputs):
    raw_predictions = model.predict(inputs)
    predictions = one_hot_encode_predictions(raw_predictions)
    accuracy = calculate_accuracy(predictions, outputs)
    return accuracy


if __name__ == "__main__":
    X, y = load_data()
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6162)

    network = Network(10)
    network.add_layer(10)
    network.add_layer(10)
    network.add_layer(5)
    network.add_layer(3)

    network.train(X_train, y_train, epochs=1000, learning_rate=0.01)
    accuracy = evaluate_model(network, X_test, y_test)
    print("Accuracy:", accuracy)
