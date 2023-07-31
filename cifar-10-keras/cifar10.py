from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import random
import tensorflow as tf

seed_value = 6162

data_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
test_file = 'test_batch'

tf.random.set_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
    return dict

### Load the Iris dataset and create training and testing sets
def load_data():
    for file in data_files:
        data_dict = unpickle("data/" + file)
        if file == data_files[0]:
            train_x = data_dict[b'data']
            train_y = data_dict[b'labels']
        else:
            train_x = np.concatenate((train_x, data_dict[b'data']), axis = 0)
            train_y = np.concatenate((train_y, data_dict[b'labels']), axis = 0)

    # Load test data 
    test_dict = unpickle("data/" + test_file)
    test_x = test_dict[b'data']
    test_y = test_dict[b'labels']

    train_x = train_x.reshape(-1, 32, 32, 3)
    test_x = test_x.reshape(-1, 32, 32, 3)

    train_y = to_categorical(train_y, num_classes=10)
    test_y = to_categorical(test_y, num_classes=10)

    return train_x, train_y, test_x, test_y

### Define the Keras model
def create_model():
    model = Sequential()
    adam = Adam(learning_rate=0.01)
    model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(10, activation = 'softmax'))


    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

### Calculate the accuracy of the CIFAR-10 prediction model
def calculate_accuracy(predictions, outputs):
    correct = 0
    for i in range(len(predictions)):
        if np.argmax(predictions[i]) == outputs[i]:
            correct += 1
    return correct / len(predictions)

### Evaluate the model
def evaluate_model(model, inputs, outputs):
    predictions = model.predict(inputs)
    accuracy = calculate_accuracy(predictions, outputs)
    return accuracy


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    print(len(X_train), len(y_train))

    model = create_model()
    model.fit(X_train, y_train, epochs=100, verbose=1)
    accuracy = evaluate_model(model, X_test, y_test)
    print("Accuracy:", accuracy)
