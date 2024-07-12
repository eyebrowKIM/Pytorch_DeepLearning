en_text = "A Dog Run back corner near spare bedrooms"

import spacy
spacy_en = spacy.load('en_core_web_sm') # spacy.load('en')은 deprecated
#오류 시 python -m spacy download en

def tokenize(en_text):
    return [tok.text for tok in spacy_en.tokenizer(en_text)]


print(tokenize(en_text))

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
print(word_tokenize(en_text))

print(en_text.split())

kor_text = "사과의 놀라운 효능이라는 글을 봤어. 그래서 오늘 사과를 먹으려고 했는데 사과가 썩어서 슈퍼에 가서 사과랑 오렌지 사왔어"

print(kor_text.split())

# from konlpy.tag import Mecab
# tokenizer = Mecab()
# print(tokenizer.morphs(kor_text))

import urllib.request
import pandas as pd
from konlpy.tag import Mecab
from nltk import FreqDist
import numpy as np
import matplotlib.pyplot as plt

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
data = pd.read_table('ratings.txt') # 데이터프레임에 저장

sample_data = data[:100] # 임의로 100개만 저장

sample_data['document'] = sample_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# 한글과 공백을 제외하고 모두 제거
sample_data[:10]




