import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

y_one_hot = torch.zeros(8, 3)
# unsqueeze -> [[2], [2], [2], [1], [1], [1], [0], [0]]
# _scatter -> dim=1, index=[[2], [2], [2], [1], [1], [1], [0], [0]], src=1
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)

# model = nn.Linear(4, 3)

class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)
        
    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()

optimizer = optim.SGD(model.parameters(), lr=0.1)

n_epochs = 1000
for epoch in range(n_epochs + 1):
    prediction = model(x_train)
    
    cost = F.cross_entropy(prediction, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, n_epochs, cost.item()
        ))