import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

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
class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.fc(x))

### Train the model
def train_model(model, dataloader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        for batch in dataloader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

### Evaluate the model
def evaluate_model(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            outputs = torch.round(model(inputs))
            total += targets.size(0)
            correct += (outputs == targets).sum().item()
    
    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    X_train, y_train = create_or_dataset(num_samples=800)
    X_test, y_test = create_or_dataset(num_samples=200)

    X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float().unsqueeze(1)
    X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float().unsqueeze(1)

    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    model = Perceptron()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1)

    train_model(model, train_loader, criterion, optimizer, epochs=100)
    accuracy = evaluate_model(model, test_loader)
    print("Accuracy:", accuracy)
