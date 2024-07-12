import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 훈련 데이터
x_train  = torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  80], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  = torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

W = torch.zeros(3, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W, b], lr=1e-5)

n_epochs = 10000

for epoch in range(n_epochs):
    
    hypothesis = x_train@W + b
    
    cost = torch.mean((hypothesis - y_train) ** 2)
    
    optimizer.zero_grad()
    
    cost.backward()
    
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
            epoch, n_epochs, hypothesis.squeeze().detach(), cost.item()
        ))