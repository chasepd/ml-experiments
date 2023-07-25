from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random


### Create an XOR dataset
def create_xor_dataset(num_samples=100):
    inputs = []
    outputs = []
    for i in range(num_samples):
        x = random.randint(0, 1)
        y = random.randint(0, 1)
        inputs.append([x, y])
        outputs.append(x ^ y)
    return inputs, outputs

### Define the model
def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

### Train the model
def train_model(model, inputs, outputs, epochs=10):
    model.fit(inputs, outputs, epochs=epochs, verbose=0)

### Evaluate the model
def evaluate_model(model, inputs, outputs):
    loss, accuracy = model.evaluate(inputs, outputs, verbose=0)
    return loss, accuracy

if __name__ == "__main__":

    train_inputs, train_outputs = create_xor_dataset(num_samples=1000)
    test_inputs, test_outputs = create_xor_dataset()

    model = create_model()
    train_model(model, train_inputs, train_outputs, epochs=100)
    loss, accuracy = evaluate_model(model, test_inputs, test_outputs)
    print("Loss:", loss)
    print("Accuracy:", accuracy)