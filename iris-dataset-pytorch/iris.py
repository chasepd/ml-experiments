import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader

# load and preprocess data
iris = load_iris()
X = iris['data']
y = iris['target']
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()

# DataLoader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32)

# define the model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(4, 10)  # Hidden layer
        self.output = nn.Linear(10, 3)  # Output layer
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

model = MLP()

# define the loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# train the model
for epoch in range(1000):
    for batch in train_loader:
        x_batch, y_batch = batch
        output = model(x_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# evaluate the model
model.eval()
with torch.no_grad():
    output = model(X_test)
    correct = (torch.argmax(output, dim=1) == torch.argmax(y_test, dim=1)).type(torch.float).sum().item()
    accuracy = correct / len(y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
