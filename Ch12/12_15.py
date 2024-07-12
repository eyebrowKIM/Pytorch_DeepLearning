import torch
import torch.nn as nn
import torch.optim as optim

sentence = "Repeat is the best medicine for memory".split()

vocab = list(set(sentence))

word2index = {tkn: i for i, tkn in enumerate(vocab, 1)} # 단어에 고유한 정수 부여
word2index['<unk>'] = 0

index2word = {v: k for k, v in word2index.items()}

def build_data(sentence, word2index):
    encoded = [word2index[token] for token in sentence] # 각 문자를 정수로 변환. 
    input_seq, label_seq = encoded[:-1], encoded[1:] # 입력 시퀀스와 레이블 시퀀스를 분리
    input_seq = torch.LongTensor(input_seq).unsqueeze(0) # 배치 차원 추가
    label_seq = torch.LongTensor(label_seq).unsqueeze(0) # 배치 차원 추가
    return input_seq, label_seq

X, Y = build_data(sentence, word2index)

class Net(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, batch_first=True):
        super(Net, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=input_size)
        
        self.rnn_layer = nn.RNN(input_size, hidden_size, batch_first=batch_first)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        #(배치 사이즈, 시퀀시 길이) -> (배치 사이즈, 시퀀스 길이, 임베딩 차원)
        output = self.embedding_layer(x)
        #(배치 사이즈, 시퀀스 길이, 임베딩 차원) -> (배치 사이즈, 시퀀스 길이, 은닉 상태 크기)
        output, hidden = self.rnn_layer(output)
        #(배치 사이즈, 시퀀스 길이, 은닉 상태 크기) -> (배치 사이즈, 시퀀스 길이, 단어 집합의 크기))
        output = self.linear(output)
        
        #(배치 사이즈, 시퀀스 길이, 단어 집합의 크기) -> (배치 사이즈 * 시퀀스 길이, 단어 집합의 크기)
        return output.view(-1, output.size(2))
    
vocab_size = len(word2index)
input_size = 5
hidden_size = 20

# 모델 생성
model = Net(vocab_size, input_size, hidden_size, batch_first=True)
# 손실함수 정의
loss_function = nn.CrossEntropyLoss() # 소프트맥스 함수 포함이며 실제값은 원-핫 인코딩 안 해도 됨.
# 옵티마이저 정의
optimizer = optim.Adam(params=model.parameters())

# 수치화된 데이터를 단어로 전환하는 함수
decode = lambda y: [index2word.get(x) for x in y]

# 훈련 시작
for step in range(201):
    # 경사 초기화
    optimizer.zero_grad()
    # 순방향 전파
    output = model(X)
    # 손실값 계산
    loss = loss_function(output, Y.view(-1))
    # 역방향 전파
    loss.backward()
    # 매개변수 업데이트
    optimizer.step()
    # 기록
    if step % 40 == 0:
        print("[{:02d}/201] {:.4f} ".format(step+1, loss))
        pred = output.softmax(-1).argmax(-1).tolist()
        print(" ".join(["Repeat"] + decode(pred)))
        print()
