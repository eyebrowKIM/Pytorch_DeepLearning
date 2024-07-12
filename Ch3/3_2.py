import torch

w = torch.tensor(2.0, requires_grad=True)

y = w**2
z = 2*y + 5

z.backward()

# z에 대해 backward 함으로써 w에 대한 미분값을 구할 수 있다.
print(w.grad)