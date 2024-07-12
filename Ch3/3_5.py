import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(1)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x)
    
class MultivariableLinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear == nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)
    
def lr_single_variable():
    # 데이터
    x_train = torch.FloatTensor([[1], [2], [3]])
    y_train = torch.FloatTensor([[2], [4], [6]])

    # 모델 초기화
    model = LinearRegressionModel()

    n_epochs = 1000

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        prediction = model(x_train)

        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()


        if epoch % 100 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, n_epochs, cost.item()
            ))
            
def lr_multiple_variable():
    # 데이터
    x_train  = torch.FloatTensor([[73,  80,  75], 
                                 [93,  88,  93], 
                                 [89,  91,  80], 
                                 [96,  98,  100],   
                                 [73,  66,  70]])  
    y_train  = torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

    # 모델 초기화
    model = MultivariableLinearRegressionModel(3)

    n_epochs = 1000

    optimizer = optim.SGD(model.parameters(), lr=1e-6)

    for epoch in range(n_epochs):
        prediction = model(x_train)

        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, n_epochs, cost.item()
            ))
            
if __name__ == '__main__':
    lr_single_variable()
    # lr_multiple_variable()