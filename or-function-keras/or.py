from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
import random


### Create an OR dataset
def create_or_dataset(num_samples=100):
    inputs = []
    outputs = []
    for i in range(num_samples):
        input = [random.randint(0, 1), random.randint(0, 1)]
        output = int(input[0] or input[1])
        inputs.append(input)
        outputs.append(output)
    return inputs, outputs

### Define the model
def create_model():
    model = Sequential()
    model.add(Dense(1, activation='sigmoid'))
    adam = Adam(learning_rate=1)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

### Train the model
def train_model(model, inputs, outputs, epochs=10):
    model.fit(inputs, outputs, epochs=epochs, verbose=0)

### Evaluate the model
def evaluate_model(model, inputs, outputs):
    loss, accuracy = model.evaluate(inputs, outputs, verbose=0)
    return loss, accuracy

if __name__ == "__main__":
    train_inputs, train_outputs = create_or_dataset(num_samples=10)
    test_inputs, test_outputs = create_or_dataset()

    model = create_model()
    train_model(model, train_inputs, train_outputs, epochs=100)
    loss, accuracy = evaluate_model(model, test_inputs, test_outputs)
    print("Loss:", loss)
    print("Accuracy:", accuracy)