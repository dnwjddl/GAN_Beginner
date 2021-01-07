- 순환 신경망으로 특정 글의 스타일을 흉내낸 텍스트 seq를 생성하는데 적용
- 문장에서 가능한 질문-대답 쌍 생성

# RNN(처음)
- 순환층은 매우 간단
- tanh() 하나로 구성(time step사이에 정보를 -1~1사이로 scale 맞춤
  - gradient Vanishing(그레이디언트 소실) : **긴 sequence 가진 데이터에 안 좋음**  
***RNN은 격차가 늘수록 학습 정보를 잃어버림***

# LSTM(Long short-term memory)
### 1) 텍스트를 정제하고 token화
✔ ```Tokenize``` : 텍스트를 단어나 문자와 같은 개별 단위로 나누는 작업  

|단어 토큰|문자 토큰|
|:-------------:|:---------:|
|텍스트를 소문자로 변환|대문자는 소문자로 바꾸거나 별도의 토큰으로 담겨두기|
|드물게 나타난거 삭제|문자의 seq를 생성해 훈련 어휘사전에 없는 새로운 단어 제작|
|stemming(단어에서 어간 추출)||
|구두점(.)와(,)를 토큰화 or 제거||
|단어 토큰화를하면 훈련 어휘사전에 없는 단어는 모델이 예측X|어휘사전은 비교적 매우 작음|
||마지막 출력 층에 학습할 가중치수가 적기 때문에 모델 훈련 속도에 유리

```python
#텍스트 정제
text = text.lower()
text = start_story + text
text = text.replace("\n\n\n\n\n", start_story)
text = text.replace("\n", " ")
text = re.sub('  +', '. ', text).strip()
text = text.replace("..",".")

text = re.sub('([!"#$%()*+,-./:;<=>?@[\]^_'{|}~])','r'\1', text)
text = re.sub('\s{2,}','', text)
```
```python
#토큰화
tokenizer = Tokenizer(char_level = False, filters = '') #단어토큰화
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1
token_list = tokenizer.texts_to_sequences([text])[0]
```

```tokenizer.fit_on_texts```: 단어 dictionary 생성  
```tokenizer.word_index```:단어 dictionary의 index 추출  
```tokenizer.texts_to_sequences```: 단어 dictionary에서 생성한 정수 인덱스로 텍스트 대체  

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences) # fit_on_texts()안에 코퍼스를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성한다.

print(tokenizer.word_index) #{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7, 'good': 8, 'knew': 9, 'driving': 10, 'crazy': 11, 'went': 12, 'mountain': 13}

print(tokenizer.word_counts) #OrderedDict([('barber', 8), ('person', 3), ('good', 1), ('huge', 5), ('knew', 1), ('secret', 6), ('kept', 4), ('word', 2), ('keeping', 2), ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)])

print(tokenizer.texts_to_sequences(sentences)) #[[1, 5], [1, 8, 5], [1, 3, 5], [9, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [7, 7, 3, 2, 10, 1, 11], [1, 12, 3, 13]]
```


### 2) 데이터셋 구축
- text의 단어들 길이는 50436
- text내 unique한 단어들의 갯수는 4169개  
- 길이가 20인 Sequence : 텍스트를 20개의 단어 길이로 나눔
- 훈련 데이터셋 X : [50416,20] 크기의 배열 (50436-20)
- 각 시퀀스의 타깃은 다음 단어 **(20벡터의 X가 4169벡터의 y와 대응)**
- 단어의 길이가 4169인 벡터로 원핫 인코딩 됨
- 타깃 y : [50416, 4169] 크기의 0,1 이진 배열

### 3) LSTM Model

```python
n_units = 256
embedding_size = 100

text_in = Input(shape = (None,))
embedding = Embedding(total_words, embedding_size)
x = embedding(text_in)
x = LSTM(n_units)(x)
x = Dropout(0.2)(x)
text_out = Dense(total_words, activation = 'softmax')(x)

model = Model(text_in, text_out)

opti = RMSprop(lr = 0.001)
model.compile(loss='categorical_crossentropy', optimizer=opti)
```

![image](https://user-images.githubusercontent.com/72767245/103923534-9002b300-5158-11eb-8dc0-e6814745c9ea.png)

