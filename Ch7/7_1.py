import numpy as np

timesteps = 10 # 시점의 수. NLP에서는 보통 문장의 길이가 된다.
input_size = 4 # 입력의 차원. NLP에서는 보통 단어 벡터의 차원이 된다.
hidden_size = 8 # 은닉 상태의 크기. 메모리 셀의 용량이다.

inputs = np.random.random((timesteps, input_size)) # 입력에 해당되는 2D 텐서. 
# (10, 4)크기의 2D 텐서

hidden_state_t = np.zeros((hidden_size,)) # 초기 은닉 상태는 0(벡터)로 초기화 
# (8, )크기의 1D 텐서 생성.
# 은닉 상태의 크기 hidden_size로 은닉 상태를 만듬.

Wx = np.random.random((hidden_size, input_size) ) # (8, 4)크기의 2D 텐서 생성. 입력에 대한 가중치.
Wh = np.random.random((hidden_size, hidden_size)) # (8, 8)크기의 2D 텐서 생성. 은닉 상태에 대한 가중치.
b = np.random.random((hidden_size,)) # (8,)크기의 1D 텐서 생성. 이 값은 편향(bias).

total_hidden_states = []

for input_t in inputs:
    output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t) + b)
    total_hidden_states.append(list(output_t))
    hidden_state = output_t

total_hidden_states = np.stack(total_hidden_states, axis=0)

print(total_hidden_states)


import torch
import torch.nn as nn

input_size = 5
hidden_size = 8

inputs = torch.Tensor(1, 10, 5)

cell = nn.RNN(input_size, hidden_size, batch_first=True)

# 양방향 순환 신경망(Bidirectional Recurrent Neural Network)
# cell = nn.RNN(input_size = 5, hidden_size = 8, num_layers = 2, batch_first = True, bidirectional = True)

print(cell(inputs))