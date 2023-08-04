from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.datasets import cifar10
import numpy as np
import random
import tensorflow as tf

seed_value = 6162

data_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
test_file = 'test_batch'

tf.random.set_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

max_epoch = 5

lr_decay = 0.993
# Define Learning Rate Scheduler
def scheduler(epoch, lr):
    if epoch < 5:
        return 0.0001
    else:
        return lr * lr_decay

lr_callback = LearningRateScheduler(scheduler)

### Load and preprocess the CIFAR-10 data
def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= np.max(X_train)
    X_test /= np.max(X_test)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return X_train, y_train, X_test, y_test

### Define the Keras model
def create_model():
    model = Sequential()
    adam = Adam(learning_rate=0.001)
    model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (5, 5), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (5, 5), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (5, 5), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(10, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

### Calculate the accuracy of the CIFAR-10 prediction model
def calculate_accuracy(predictions, outputs):
    correct = 0
    for i in range(len(predictions)):
        if np.argmax(predictions[i]) == np.argmax(outputs[i]):
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

    early_stopping = EarlyStopping(
        monitor='val_loss', 
        min_delta=0.001,    
        patience=50,        
        restore_best_weights=True,
    )
    model = create_model()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=1024, callbacks=[lr_callback, early_stopping], verbose=1)
    accuracy = evaluate_model(model, X_test, y_test)
    print("Accuracy:", accuracy)
