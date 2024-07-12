import torch
import torch.optim as optim

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

W, b = torch.zeros(1), torch.zeros(1)

n_epochs = 1000

# Optimizer를 이용한 구현
# for epoch in range(n_epochs):
#     hypothesis = x_train * W + b

#     cost = torch.mean((hypothesis - y_train) ** 2)
#     print(cost)

#     optimizer = optim.SGD([W, b], lr=0.01)

#     #optimizer의 grad를 0으로 초기화
#     optimizer.zero_grad()
#     #비용함수를 미분하여 gradient 계산
#     cost.backward()
#     #optimizer.step()을 호출하면 W와 b가 업데이트
#     optimizer.step()
#     print(W, b)
    
#     if epoch % 100 == 0:
#         print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
#             epoch, n_epochs, W.item(), b.item(), cost.item()
#         ))
        

lr = 0.0001
# Gradient Descent를 이용한 구현
while True:
    hypothesis = x_train * W + b
    
    cost = torch.mean((hypothesis - y_train) ** 2)
    
    # gradient 계산
    gradient = torch.sum((W * x_train - y_train) * x_train)
    # 업데이트
    W -= lr * gradient
    b -= lr * torch.sum(W * x_train - y_train)
    
    if cost < 0.01:
        break

print(W, b)