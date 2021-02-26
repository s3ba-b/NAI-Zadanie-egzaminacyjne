# Imports
import torch.nn as nn
import numpy as np
import torch


class SimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(13, 3)
        self.act1 = nn.ReLU()  # Activation function
        self.linear2 = nn.Linear(3, 1)

    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x


inputs = np.loadtxt("../data/housing.data", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), dtype='float32')
targets = np.loadtxt("../data/housing.data", usecols=13, dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# Import tensor dataset & data loader
from torch.utils.data import TensorDataset, DataLoader

# Define dataset
train_ds = TensorDataset(inputs, targets)
train_ds[0:3]

# Define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
next(iter(train_dl))

# Define model
model = SimpleNet()
# print(model.weight)
# print(model.bias)

# Define optimizer
opt = torch.optim.SGD(model.parameters(), 1e-5)

# Import nn.functional
import torch.nn.functional as F

# Define loss function
loss_fn = F.mse_loss


# Define a utility function to train the model
def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
    print('Training loss: ', loss_fn(model(inputs), targets))


# Train the model for 100 epochs
fit(100, model, loss_fn, opt)

# Generate predictions
preds = model(inputs)
preds

# Compare with targets
targets
