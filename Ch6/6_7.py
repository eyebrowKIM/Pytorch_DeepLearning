import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import MNIST


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)

if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    
mnist_train = MNIST(root='.data/',
                    train=True,
                    transform=transforms.ToTensor(),
                    download=True)
mnist_test = MNIST(root='.data/',
                     train=False,
                     transform=transforms.ToTensor(),
                     download=True)

batch_size=100

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

model = nn.Sequential(
    nn.Linear(784, 10, bias=True),
    nn.ReLU(),
    nn.Linear(10, 10, bias=True),
    nn.ReLU(),
    nn.Linear(10, 10, bias=True),
    nn.ReLU(),
    nn.Linear(10, 10, bias=True),
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

n_epochs = 15

for epoch in range(n_epochs):
    for batch_idx, samples in enumerate(data_loader):
        x_train, y_train = samples
        x_train = x_train.view(-1, 28*28).to(device)
        y_train = y_train.to(device)
        
        optimizer.zero_grad()
        prediction = model(x_train)
        
        cost = loss_fn(prediction, y_train)
        
        cost.backward()
        optimizer.step()
        
        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, n_epochs, batch_idx+1, len(data_loader),
            cost.item()))