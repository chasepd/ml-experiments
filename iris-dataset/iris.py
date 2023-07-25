from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from neural_net import Network


### Load the Iris dataset and create training and testing sets
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6162)

    network = Network(4)
    network.add_layer(4, activation_function="sigmoid")
    network.add_layer(5, activation_function="sigmoid")
    network.add_layer(10, activation_function="sigmoid")
    network.add_layer(3, activation_function="sigmoid")

    network.train(X_train, y_train, epochs=1000, learning_rate=0.1)
    accuracy = evaluate_model(network, X_test, y_test)
    print("Accuracy:", accuracy)
