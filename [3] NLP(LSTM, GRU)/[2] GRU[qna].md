## 적층 순환 네트워크 (LSTM을 여러층)
- 첫번째 LSTM층의 ```return_sequences```을 True 로 반환, 순환 층이 마지막 타임스텝의 은닉 상태만 출력하지 않고 모든 타임스텝의 은닉 상태를 출력
- 두번째 LSTM층은 첫번째 은닉 상태를 입력 데이터로 사용

```python
text_in = Input(shape= (None,))
embedding = Embedding(total_words, embedding_size)

x = embedding(text_in)
x = LSTM(n_units, return_sequence = True)(x)
x = LSTM(n_units)(x)
x = Dropout(0.2)(x)
text_out = Dense(total_words, activation= 'softmax')(x)

model = Model(text_in, text_out)
```

# GRU(gated Recurrent Unit)
**차이점**
- 삭제 게이트와 입력게이트가 리셋 게이트와 업데이트 게이트로 바뀐다
- 셀 상태와 출력 게이트가 없다. 셀은 은닉 상태만 출력

![image](https://user-images.githubusercontent.com/72767245/104047799-9743c280-5225-11eb-83ee-fd8ab2c33958.png)

## workflow
- Reset Gate: 이전 히든 스테이트 h_(t-1)과 현재 입력 값 x_t를 고려해, 현재 입력 값을 히든 스테이트 h_t에 얼마나 반영할 지 판단

![image](https://user-images.githubusercontent.com/72767245/104048261-497b8a00-5226-11eb-99b8-b17c33a3e1d5.png)

- Update Gate: 히든 스테이트 h_(t-1)과 입력 값 x_t로부터 z_t값을 생성하고, z_t를 기준으로 Reset Gate로 부터 반환된 값과 이전 hidden state중 어디에 얼만큼 가중치를 둘지 판단

![image](https://user-images.githubusercontent.com/72767245/104048266-4c767a80-5226-11eb-8eb0-ab3437474432.png)

## 양방향 셀(Bidirectional Layer)
- inference time(훈련된 모델을 사용하여 예측을 만드는 것, 추론시) 전체 텍스트를 모델에 제공할 수 있는 예측 문제에서는 시퀀스를 전진방향으로만 처리할 이유가 없다.(후진방향으로도 처리 가능)
- 두개의 은닉상태를 사용
  - 하나는 일반적인 전진 방향으로 처리되는 시퀀스의 결과를 저장
  - 다른 하나는 시퀀스가 후진 방향으로 처리될 때 만들어짐
- 순환층은 주어진 타임스텝의 앞과 뒤에서 모두 학습 가능

```python
layer = Bidirectional(GRU(100))
# 은닉상태의 길이는 200인 벡터가 됨
```

# Encoder-Decoder Model
- **hidden state**, 층이 가진 현재 시퀀스에 대한 지식
- 마지막 은닉 상태를 fully connected Layer에 연결하면 네트워크가 다음 단어에 대한 확률 분포를 출력
- 단어 하나를 예측하는 것이 아니며, 입력 시퀀스에 관련된 완전히 다른 단어의 시퀀스를 예측해야 함

## 실생활 적용
```언어 번역```  
네트워크에 source 언어로 된 텍스트를 주입하고 target 언어로 번역된 텍스트를 출력  
```질문 생성```  
네트워크에 텍스트 문장을 주입하고 텍스트에 관해 가능한 질문을 생성하는 것이 목적  
```텍스트 요약```  
네트워크에 긴 텍스트 문장을 주입하고 이 문장의 짧은 요약을 생성하는 것이 목적

### 순차 데이터에서 ENCODER-DECODER 과정
- 원본 입력 Sequence는 ENCODER RNN에 의해 하나의 벡터로 요약 (전체 입력 문서에 대한 하나의 표현)
- 이 벡터는 DECODER RNN의 초깃값으로 사용
- 각 time step에서 DECODER RNN의 은닉상태는 fully-connected Layer에 연결되어 단어 어휘사전에 대한 확률 분포를 출력
  - 이런식으로 인코더가 생성한 입력 데이터로 디코더를 초기화한 다음 새로운 Text Sequence 생성 가능
<br>

- **Loss Function** : 각 time step 에서 디코더가 생성한 출력 분포를 진짜 다음 단어와 비교하여 손실을 계산  
  - 훈련 과정에서 디코더가 이 분포로부터 샘플링하여 다음 단어를 생성할 필요는 없음
  - 이전 출력 분포에서 단어를 샘플링하는 대신 그 다음 time step에 진짜 다음 단어 cell이 주입되기 때문
  <br>
인코더 디코더 네트워크의 이런 훈련 방식을 **Teacher Forcing** 이라 함
  - 예측은 하되 강제로 정답을 예측하게끔 훈련
  
# 질문-대답 생성기
- RNN 하나가 text로 부터 대답 후보를 고른다
- Encoder-Decoder network가 RNN이 선택한 대답 후보 중 하나가 주어졌을 때 적절한 질문을 생성
- ```첫번째 NETWORK```: '가능성' 있는 대답을 찾을 수 있다
- ```두번째 NETWORK```: '각각의 대답에 대한 질문 생성
## 분석
- 데이터의 열 분석

|열 이름|설명|
|:-------:|-------------------|
|story_id|스토리에 대한 고유 식별자|
|story_text|스토리의 텍스트|
|question|스토리 텍스트에 대한 질문|
|answer_token_ranges|스토리 텍스트에서 대답의 토큰 위치|
|document_tokens|토큰화된 스토리 텍스트|
|question_input_tokens|토큰화된 질문|
|question_output_tokens|한 타임스텝 밀린 토큰화된 질문|
|answer_masks|[max_answer_length, max_document_length] 크기의 이진 마스크 행렬|
|answer_labels|max_document_length길이의 이진벡터|


## 모델
입력: ```document_tokens```  
양방향 셀-> ```answer_masks```    
(손실)```answer_labels```   
인코더 입력: ```answer_masks```  
인코더   
디코더 입력: ```question_input_tokens```  
디코더   
(손실) ```question_output_tokens```  

```python

#### 질문-대답 쌍을 생성하기 위한 모델 구조 ####

from keras.layers import Input, Embedding, GRU, Bidirectional, Dense, Lambda
from keras.models import Model, load_model
import keras.backend as K
from qgen.embedding import glove

### 파라미터

VOCAB_SIZE = glove.shape[0] # 9984
EMBEDDING_DIMENS = glove.shape[1] # 100

GRU_UNITS = 100
DOC_SIZE = None
ANSWER_SIZE = None
Q_SIZE = None

document_tokens = Input(shape = (DOC_SIZE,), name = 'document_tokens')

# 임베딩 층은 GloVe단어 벡터로 초기화
embedding = Embedding(input_dim = VOCAB_SIZE, output_dim = EMBEDDING_DIMENS, weight = [glove], mask_zero= True, name = 'embedding')
document_emb = embedding(document_tokens)

# 순환층은 각 타임스텝의 은닉상태를 반환하는 양방향 GRU
answer_outputs = Bidirectional(GRU(GRU_UNITS, return_sequences = True), name = 'answer_outputs')(document__emb)
# 출력 Dense 층은 각 타임스텝의 은닉상태에 연결된다
# 이 층은 두개의 유닛과 소프트 맥스 활성화 함수로 이루어져 있음
# 각 단어가 대답의 일부분인지(1) 또는 대답의 일부분이 아닌지(0)에 대한 확률 포함
answer_tags = Dense(2, activation = 'softmax', name = 'answer_tags')(answer_outputs)
```
```python
#### 대답이 주어졌을때 질문을 구성하는 인코더-디코더 네트워크의 모델 ####
encoder_input_mask = Input(shape = (ANSWER_SIZE, DOC_SIZE), name = "encoder_input_mask")
encoder_inputs = Lambda(lambda x: K.batch_dot(x[0], x[1]), name = "encoder_inputs")([encoder_input_mask, answer_outputs])
encoder_cell = GRU(2*GRU_UNITS, name = 'encoder_cell')(encoder_inputs 

decoder_inputs = Input(shape = (Q_SIZE,), name = 'decoder_inputs')
decoder_emb = embedding(decoder_inputs)
decoder_emb.trainable = False
decoder_cell = GRU(2*GRU_UNITS, return_sequences = True, name = 'decoder_cell')
decoder_states = decoder_cell(decoder_emb, initial_state = [encoder_cell])

decoder_projection = Dense(VOCAB_SIZE, name = 'decoder_projection', activation = 'softmax', use_bias = False)
decoder_outputs = decoder_projection(decoder_states)

total_model = Model([document_tokens, decoder_inputs, encoder_input_mask], [answer_tags, decoder_outputs])
answer_model = Model(documents_tokens, [answer_tags])
decoder_initial_state_model = Model([document_tokens, encoder_input_mask], [encoder_cell])

```

## 예측
- 예측을 할때는 Teacher Forcing을 해주지 않는다
## 모델 결과
- 인코더가 가능한 대답으로부터 문맥을 추출하기 때문에 디코더는 이에 맞는 질문을 생성가능
- 디코더는 <UNK>태그로 마쳤다
   - 무엇이 나올지 몰라서가 아니고 어휘사전에 없는 단어를 예측하였기 때문
- 모델의 정확도와 생성 능력을 향상시키기 위해 인코더-디코더 네트워크를 확장한 것이 많이 있다
  - Poiter Network: 어휘사전에 있는 단어에만 의존하는 것이 아니라 모댈이 생성된 질문에 포함할 입력 텍스트의 특정 단어를 'pointing' 할 수 있음 (<UNK> 문제 해결)
  - Attention Mechanism

