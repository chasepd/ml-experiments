import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from neural_net import Network
import random

seed_value = 6162
random.seed(seed_value)
np.random.seed(seed_value)

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
