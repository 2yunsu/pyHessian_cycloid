import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.init as init
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
import pdb

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

#random seed
random_seed = 0
torch.manual_seed(random_seed)  # torch
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
np.random.seed(random_seed)  # numpy
random.seed(random_seed)  # random

#parameter
num_epochs = 10000

#cycloid
theta_1 = torch.linspace(0, 2 * np.pi, num_epochs).view(-1,1)
r = 1
# Define the input and target
x = torch.linspace(1, 1, 1).view(-1,1)
target = torch.linspace(np.pi, np.pi, 1).view(-1,1)

# Define the model
class RegNN(nn.Module):
    def __init__(self):
        super(RegNN, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

def cycl_loss(output, target):
    theta_1 = torch.linspace(0, 2 * np.pi, 1000).view(-1,1)
    r = 10
    x = r * (theta_1 - np.sin(theta_1))
    y = -r * (1 - np.cos(theta_1))

    min_distance = float('inf')
    closest_element = None
    for element in range(len(x)):
        distance = torch.abs(x[element] - output)
        if distance < min_distance:
            min_distance = distance
            closest_element = element

    y_cycl = y[closest_element]   

    return y_cycl


model = RegNN()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0)
criterion = torch.nn.MSELoss()
model.train()

# Fix the bias
model.linear.bias.data.fill_(0)
model.linear.bias.requires_grad = False

losses = []
crr_weights_list = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    # weight 추적
    weights = model.linear.weight.data
    bias = model.linear.bias.data
    # Forward pass
    output = model(x)
    target = target.view(-1,1)
    loss = criterion(output, target)
    # Backward pass
    loss.backward()
    crr_weights_list.append(weights.item())
    optimizer.step()

    losses.append(loss.item())

# change the model to eval mode to disable running stats upate
model.eval()
# Define the test input and target
x_test = torch.linspace(-2, 2, 100).view(-1,1)
target_test = 2 * x_test ** 3 - 4 * x_test ** 2 + 3 * x_test + torch.randn_like(x_test)

# Forward pass on the test data
output_test = model(x_test)
target_test = target_test.view(-1,1)

# Calculate the loss on the test data
loss_test = criterion(output_test, target_test)

# Print the test loss
print('Test loss:', loss_test.item())
print('Weight:', weights.item())

# Create a plot
plt.figure(figsize=(8,6))
plt.scatter(crr_weights_list, losses, label='Target data', color='blue')
# plt.scatter(x, target, label='Cycloid', color='red')
plt.legend()
plt.savefig('train_result.png')