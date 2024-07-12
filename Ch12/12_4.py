import re
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize

from tqdm import tqdm

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from konlpy.tag import Okt

# 데이터 다운로드
# urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml", filename="ted_en-20160408.xml")

# targetXML = open('ted_en-20160408.xml', 'r', encoding='UTF8')
# target_text = etree.parse(targetXML)

# # xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
# parse_text = '\n'.join(target_text.xpath('//content/text()'))

# # 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# # 해당 코드는 괄호로 구성된 내용을 제거.
# content_text = re.sub(r'\([^)]*\)', '', parse_text)

# # 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
# sent_text = sent_tokenize(content_text)

# # 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
# normalized_text = []
# for string in sent_text:
#      tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
#      normalized_text.append(tokens)

# # 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
# result = [word_tokenize(sentence) for sentence in normalized_text]

# # vector_size : 임베딩 벡터의 차원
# # window : 컨텍스트 윈도우 크기
# # min_count : 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습하지 않는다.)
# # workers : 학습을 위한 프로세스 수
# # sg : 0은 CBOW, 1은 Skip-gram.

# model = Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)
# model_result = model.wv.most_similar("man")
# print(model_result)

# model.wv.save_word2vec_format('eng_w2v')

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")

train_data = pd.read_table('ratings.txt')

train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# 형태소 분석기 OKT를 사용한 토큰화 작업 (다소 시간 소요)
okt = Okt()

tokenized_data = []
for sentence in tqdm(train_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    tokenized_data.append(stopwords_removed_sentence)
    
# 리뷰 길이 분포 확인
print('리뷰의 최대 길이 :',max(len(review) for review in tokenized_data))
print('리뷰의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))
plt.hist([len(review) for review in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()


from gensim.models import Word2Vec

model = Word2Vec(sentences = tokenized_data, vector_size = 100, window = 5, min_count = 5, workers = 4, sg = 0)

# 완성된 임베딩 매트릭스의 크기 확인
model.wv.vectors.shape

print(model.wv.most_similar("최민식"))

print(model.wv.most_similar("히어로"))
