- 순환 신경망으로 특정 글의 스타일을 흉내낸 텍스트 seq를 생성하는데 적용
- 문장에서 가능한 질문-대답 쌍 생성

# RNN(처음)
- 순환층은 매우 간단
- tanh() (or sigmoid) 하나로 구성(time step사이에 정보를 -1~1사이로 scale 맞춤
  - gradient Vanishing(그레이디언트 소실) : **긴 sequence 가진 데이터에 안 좋음**  
***RNN은 격차가 늘수록 학습 정보를 잃어버림***

# LSTM(Long short-term memory)
- cell state 추가
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
#하이퍼파라미터
n_units = 256
embedding_size = 100

#초기화
text_in = Input(shape = (None,))
embedding = Embedding(total_words, embedding_size)

#모델 구축
x = embedding(text_in)
x = LSTM(n_units)(x)
x = Dropout(0.2)(x)
text_out = Dense(total_words, activation = 'softmax')(x)

#모델 객체
model = Model(text_in, text_out)

#컴파일
opti = RMSprop(lr = 0.001)
model.compile(loss='categorical_crossentropy', optimizer=opti)
```

![image](https://user-images.githubusercontent.com/72767245/103923534-9002b300-5158-11eb-8dc0-e6814745c9ea.png)

#### Embedding 층
- 모델이 역전파를 통해 업데이트가 가능한 단어 표현을 학습할 수 있기 때문에 각 정수 토큰이 **연속적인 벡터로 임베딩**
- **입력 토큰**을 원-핫 인코딩을 할 수도 있지만 임베딩 층이 더 선호 (y값은 이미 원-핫 인코딩 함)
- 임베딩 층은 스스로 학습할 수 있기 때문에 성능이 향상시키기 위해 모델이 토큰의 임베딩 방법을 자유롭게 결정할 수 있기 때문  

시퀀스(20) * 임베딩(100)의 크기의 벡터가 입력으로 [x1, x2, x3... x20] 

#### LSTM 층
- 시퀀스(20) * 임베딩(100)의 크기의 벡터가 입력으로 [x1, x2, x3... x20]  
- 같은 층(layer)마다 같은 가중치를 공유  
<br>

- 최종 은닉 상태가 이 층의 출력, 이 벡터의 길이는 셀에 있는 유닛의 개수와 동일
  - 층을 정의할 때 지정해야되는 하이퍼파라미터
  - 시퀀스의 길이와 무상관

✔ 순환층, 셀과 유닛
- LSTM층에는 하나의 셀이 있고 이 셀은 여러 개의 유닛을 포함
- 셀을 펼쳐서 연결하면 순환 층

#### LSTM 셀
- h_t(단기 상태(short-term state)), c_t(장기 상태(long-term state))

![image](https://user-images.githubusercontent.com/72767245/103927355-95163100-515d-11eb-8270-b70bb9e5e428.png)

- Forget Gate: f_t에 의해 제어, 장기 상태 c_t-1의 어느 부분을 **삭제**할지 제어
- Input Gate: i_t에 의해 제어, g_t의 어느 부분이 장기상태 c_t-1에 **더해져야 하는지** 제어(c_t가 됨)
- Output Gate: o_t는 장기상태 c_t의 어느 부분을 읽어서 h_t와 y_t로 출력해야 하는지 제어

<img src="https://user-images.githubusercontent.com/72767245/103927728-2dacb100-515e-11eb-8745-6dc9a1375f40.png" width = 30%>

### 4) 새로운 텍스트 생성
- 기존 단어의 시퀀스를 네트워크에 주입하고 다음 단어를 예측
- 이 단어를 기존 시퀀스에 추가하고 과정을 반복  
```NETWORK```는 **샘플링할 수 있는 각 단어의 확률** 출력
- 결정적이지 않고 확률적으로 텍스트 생성
- ```temperature 매개변수```를 사용하여 샘플링 과정을 얼마나 결정적으로 만들지 지정 가능

```python
## temperature가 0에 가까울수록 샘플링을 더 결정적으로 만듦 (가장 높은 확률을 가진 단어가 선택될 가능성이 높음)
## temperature 값이 1에 가까우면 모델이 출력한 확률에 따라 단어가 선택

def sample_with_temp(pred, temperature=1.0):
  #확률 배열에서 인덱스 하나를 샘플링하는 헬퍼 함수
  
  # preds: LSTM 네트워크의 마지막 Dense 층의 소프트 맥스 활성화 함수가 만든 0~1사이의 확률
  preds = np.asarray(preds).astype('float32') 
  # 이 값을 로그 취하면 0에 가까울 수록 아주 큰 음수
  # 이를 0에 가까운 temperature로 나누고 다시 지수함수로 복원하면 작았던 확률이 더 크게 작아짐
  preds = np.log(preds)/temperature
  
  ## 가장 높았던 확률을 가진 단어가 선택될 가능성이 높아진다
  # np.random.multionmial 함수를 사용하기 위해 소프트 맥스 함수를 다시 적용하여 확률의 합을 1로 만듦
  
  # softmax 함수
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  
  prods = np.random.multionmial(1, preds, 1)
  return np.argmax(prods)
```
```python
# seed_text: 생성과정을 시작하기 위해 모델에 전달할 단어 시퀀스(공백도 가능)
# start_story => 문자블럭
def generate_text(seed_text, next_words, model, max_sequence_len, temp):
  output_text = seed_text
  seed_text = start_story + seed_text
  
  for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0] #단어 토큰의 리스트로 반환
   
    # 시퀀스가 길어질수록 다음 단어를 생성하는데 시간이 오래 걸리기 때문에 둔 시퀀스 길이 제한
    token_list = token_list[-max_sequence_len:] #마지막 max_sequence_len개의 토큰만 유지
    token_list = np.reshape(token_list, (1, max_sequence_len))
    
    # 모델은 시퀀스의 다음 단어에 대한 확률을 출력
    prods = model.predict(token_list, verbose = 0)[0]
    # 다음 단어를 출력하기 위해 샘플링 함수에 확률과 temperature 매개변수를 전달
    y_class = sample_with_temp(prods, temperature = temp)
    
    output_word = tokenizer.index_word[y_class] if y_class > 0 else ''
    
    # 출력 단어가 스토리의 시작 토큰이면 다시 시작
    if output_word == "|":
      break
    # 그렇지 않으면 새로운 단어를 seed_text에 덧붙이고 다음 생성 과정을 반복
    seed_text += output_word + ' '
    output_text += output_word + ' '
    
 return output_text
```

#### 결과
- temperature = 0.2로 생성한 텍스트가 temperature = 1.0으로 생성한 텍스트보다 덜 모험적이지만 더 논리적이다.
- temperature가 낮을수록 더 결정적인 샘플링이 되기 때문
<br><br>
- 둘 다 스토리가 여러 문장에 걸쳐 잘 이어지지 않음
  - 단어의 의미를 알지 못하기 때문
    - 사용자가 다음 시퀀스에 나올 확률이 가장 높은 단어 10개 중에 고르는 방식
