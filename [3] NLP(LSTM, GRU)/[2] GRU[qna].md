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
