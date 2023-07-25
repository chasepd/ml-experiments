from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
import numpy as np
import random
import tensorflow as tf

seed_value = 6162

tf.random.set_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

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

### Define the Keras model
def create_model():
    model = Sequential()
    adam = Adam(learning_rate=0.1)
    model.add(Dense(4, input_dim=4, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_value)

    model = create_model()
    model.fit(X_train, y_train, epochs=1000, verbose=0)
    accuracy = evaluate_model(model, X_test, y_test)
    print("Accuracy:", accuracy)
