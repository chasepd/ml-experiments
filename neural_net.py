import numpy as np

### Class to represent a single neuron
class Neuron:
    def __init__(self, num_inputs, activation_function, derivative_function):
        # Initialize weights and bias to small random numbers

        # Xavier initialization
        self.weights = np.random.randn(num_inputs) / np.sqrt(num_inputs)
        self.bias = 0
        self.activation_function = activation_function
        self.derivative_function = derivative_function

    def feedforward(self, inputs):
        # Save inputs for use in backpropagation
        self.previous_inputs = inputs
        # Weight inputs, add bias, then use the activation function
        self.last_total = np.dot(inputs, self.weights) + self.bias
        self.last_output = self.activation_function(self.last_total)
        return self.last_output
    
    def backpropagate(self, error, learning_rate):
        # Apply the chain rule to calculate derivative of the loss with respect to weights and bias
        dtotal = error * self.derivative_function(self.last_total)
        
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
    def __init__(self, num_neurons, num_inputs, activation_function, derivative_function):
        self.neurons = [Neuron(num_inputs, activation_function, derivative_function) for i in range(num_neurons)]
        self.activation_function = activation_function
        self.derivative_function = derivative_function
        
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

    def add_layer(self, num_neurons, layer_type=FullyConnectedLayer, activation_function="relu"):
        num_inputs = self.num_inputs if len(self.layers) == 0 else len(self.layers[-1].neurons)
        derivative_function = sigmoid_derivative

        if activation_function.lower() == "sigmoid":
            activation_function = sigmoid
            derivative_function = sigmoid_derivative
        elif activation_function.lower() == "relu":
            activation_function = relu
            derivative_function = relu_derivative
        elif activation_function.lower() == "leakyrelu" or activation_function.lower() == "leaky_relu" or activation_function.lower() == "leaky-relu":
            activation_function = leaky_relu
            derivative_function = leaky_relu_derivative
        elif activation_function.lower() == "tanh":
            activation_function = tanh
            derivative_function = tanh_derivative
        elif activation_function.lower() == "prelu":
            activation_function = prelu
            derivative_function = prelu_derivative
        elif activation_function.lower() == "swish":
            activation_function = swish
            derivative_function = swish_derivative
        elif activation_function.lower() == "softmax":
            activation_function = softmax
            derivative_function = dummy_derivative
        elif activation_function.lower() == "elu":
            activation_function = elu
            derivative_function = elu_derivative
        
        self.layers.append(layer_type(num_neurons, num_inputs, activation_function, derivative_function))
    
    def train(self, inputs, expected_outputs, epochs=10, learning_rate=0.01, loss_function = "cross_entropy", verbose=True):
        if loss_function.lower() == "cross_entropy":
            loss_function = softmax_cross_entropy_with_logits if self.layers[-1].activation_function == softmax else cross_entropy_loss
        elif loss_function.lower() == "binary_cross_entropy":
            loss_function = binary_cross_entropy_loss
        if verbose:
            print("Training...")
        for epoch in range(epochs):            
            total_loss = 0
            num_inputs = 0
            for single_input, single_output in zip(inputs, expected_outputs):
                outputs = self.feedforward(single_input)
                loss = loss_function(outputs, single_output)
                total_loss += loss
                num_inputs += 1
                errors = np.array(outputs) - np.array(single_output)
                for layer in reversed(self.layers):
                    errors = layer.backpropagate(errors, learning_rate)
            if verbose:
                print("Epoch", epoch + 1, "Loss:", total_loss / num_inputs)




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

### Sigmoid activation function    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

### Derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

### ReLU activation function
def relu(x):
    return np.maximum(0, x)

### Derivative of the ReLU function
def relu_derivative(x):
    return (x > 0).astype(float)

### Leaky ReLU activation function
def leaky_relu(x):
    return np.maximum(0.01 * x, x)

### Derivative of the Leaky ReLU function
def leaky_relu_derivative(x):
    dx = np.ones_like(x)
    dx[x < 0] = 0.01
    return dx

### PRELU activation function
def prelu(x, alpha):
    return np.where(x > 0, x, alpha * x)

### Derivative of the PRELU function
def prelu_derivative(x, alpha):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

### Tanh activation function
def tanh(x):
    return np.tanh(x)

### Derivative of the tanh function
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

### Swish activation function
def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    return swish(x) + sigmoid(x) * (1 - swish(x))

### Softmax activation function
def softmax(x):
    x -= np.max(x)  # For numerical stability
    return np.exp(x) / np.sum(np.exp(x), axis=0)

### Dummy derivative function for softmax
def dummy_derivative(x):
    return x 

### ELU activation function
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

### Derivative of the ELU function
def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, elu(x, alpha) + alpha)

### Cross entropy loss function
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / len(y_true)

def cross_entropy_loss_derivative(y_true, y_pred):
    return y_pred - y_true

### Binary cross entropy loss function
def binary_cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_loss_derivative(y_true, y_pred):
    return (1 - y_true) / (1 - y_pred + 1e-9) - y_true / (y_pred + 1e-9)