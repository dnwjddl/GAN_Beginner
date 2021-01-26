# 음악창작
- 모델이 순차적인 음악의 구조를 학습하고 재생성할 수 있어야 함
- **이산적인 확률 집합**을 사용하여 **연속적인 악보**를 만들 수 있어야 함
#### 고려해야할 요소
- 피치(pitch): 음, 높이
- 리듬(rhythm): 코드, 화음
- 선율(polyphonic)
  - 텍스트 생성은 하나의 텍스트 스트림만 처리하면 되지만 음악은 화음의 스트림을 동시에 처리해야 함
#### Attention Mechanism
텍스트 생성에 사용한 여러가지 RNN 기술을 음악생성에서도 사용 가능(+Attention Mechanism)  
이를 사용하면 RNN 이 다음에 나올 음표를 예측하기 위해 **이전의 어느 음표에 초점을 맞출지** 선택 가능

## 하나의 선율을 가진 음악 생성에 초점을 맞추어 문제를 단순화

## 준비사항
```python
from music21 import converter

dataset_name = 'cello' # 이 음악은 하나의 악기(첼로)만을 사용하여 연주 -> 파트 나눌 필요 없음
# 하지만 여러악기를 사용하는 음악을 생성하려면 파트를 나누어 놓는 것이 좋다
filename = 'cs1-2all'
file = "./data/{}/{}.mid".format(dataset_name, filename)

# chordify 메서드: 여러 파트로 나누어진 음표를 하나의 파트에서 동시에 연주되는 화음으로 압축
original_score = converter.parse(file).chordify() 
```
```python
# 음악기호
original_score.show()
original_score.show('text')
```
- 미디 파일은 악기, 박자, 키(key) 같은 메타데이터로 시작
- 이 작업에서는 이런 정보를 사용하지 않는다
- `{4.0} <music21.chord.Chord G2 D3 B3>` `{5.0} <music21.chord.Chord B3>`
  - 이 음표는 음악의 4번째 박자(0부터 시작)에서 시작하고 다음 음표가 5번째 박자에서 시작하므로 1박자 길이를 가짐
  - 낮은 G, D, B 코드로 구성되어 있음 
- `{6.0} <music21.chord.Chord G3>` `{6.25} <music21.chord.Chord D3>`
  - 이 음표는 6번째 박자부터 시작하고 다음음표가 6.25 부터 시작하므로 1/4박자 길이를 가지고 G 코드 하나로 구성
- `{7.75} <music21.chord.Chord C4>` `{8.0} <music21.chord.Chord D4>`
  - 이 음표는 7.75번째 박자로부터 시작하고 1/4박자 길이를 가진다. 높은 C 코드 하나로 구성

✔ 데이터 추출  
악보를 순회하며 각 음표(와 쉼표)의 피치와 박자를 두 개의 리스트로 추출  
코드 전체는 하나의 문자열로 저장되고 코드의 개별 음표는 점으로 구분한다  
- 각 음표의 이름 뒤에 있는 숫자는 음표가 속한 옥타브를 지칭  
  - 동일한 음표 이름(A~G:코드)이 반복되기 때문에 고유한 피치를 구분하기 위해 필요  
    - G2는 G3보다 한 옥타브 낮음

```python
### 데이터 추출 ###
notes = []
durations = [] #음이 지속되는 길이(박자)

for element in original_score.flat:
  if isinstance(element, chord.Chord):
    notes.append('.'.join(n.nameWithOctave for n in element.pitches))
    durations.append(element.duragion.quarterLength)
    
  if isinstance(element, note.Note):
    if element.isRest:
      notes.append(str(element.name))
      durations.append(element.duration.quarterLength)
    else:
      notes.append(str(element.nameWithOctave))
      durations.append(element.duration.quarterLength)
```
✔ isinstance함수  
인스턴스가 특정 클래스/데이터 타입과 일치한지 알아봐주는 함수  
<br>
- 단어는 하나의 ```pitch``` (높 낮이, 음)
- ```pitch```에 시퀀스가 주어지면 다음 pitch를 예측하는 모델을 만들어야 함
## 첫번째 음악생성 RNN
- 전처리: 피치&박자를 정숫값으로 변환
- 데이터를 32개의 음표씩 나누어 훈련세트 만듦
- 타깃은 시퀀스에 있는 (원-핫 인코딩된) 다음 피치와 박스  
<br>
`pitch input`: [386, 77, 340...]  
`duration input`: [0,3,8,3,3,...]  
`pitch output`: [0.,0.,0.,0.,0.,...]    
`duration output`: [0.,0.,1.,0.,...]  
<br>
- 복잡한 순차 생성 모델에서 필수가 된 어텐션 메커니즘
- 이 알고리즘은 순환층이나 합성곱 층이 필요하지 않고 완전히 어텐션만으로 구성된 ```Transformer 모델```을 탄생


### Attention
어텐션 메커니즘은 원래 텍스트 번역 문제, 특히 영어문장을 프랑스어로 번역하는 문제에 적용된 모델
##### `AutoEncoder`  
- z: 문맥을 담은 벡터 생성, 문맥벡터가 병목될 가능성 있음
- 긴 문장의 시작 부분에 정보는 문맥벡터에 도달할 때 희석될 수 있음  
##### `Attention Mechanism`
- 어텐션 메커니즘에서는 모델인 인코더 RNN의 마지막 은닉 상태만 문맥 벡터로 사용하지 않고 인코더 RNN의 이전 타임스텝에 있는 ```은닉 상태의 가중치 합```으로 문맥 벡터를 만듦
- ```어텐션 메커니즘```은 인코더의 이전 은닉상태와 디코더의 현재 은닉상태를 문맥 벡터 생성을 위한 덧셈 가중치로 변환하는 일련의 층  
  
1. 간단한 순환 층 뒤에 어텐션을 추가하는 방법    
2. 인코더-디코더 네트워크로 확장  
3. 음표 하나가 아니라 전체 음표 시퀀스 예측  

### Keras로 Attention Mechanism 생성
#### 기존 RNN
![https://user-images.githubusercontent.com/72767245/105803721-631c2000-5fe1-11eb-9814-68f03f7ee95b.png](https://user-images.githubusercontent.com/72767245/105803721-631c2000-5fe1-11eb-9814-68f03f7ee95b.png)

#### Attention Mechanism with RNN
![image](https://user-images.githubusercontent.com/72767245/105805793-1edf4e80-5fe6-11eb-95a1-7550fc9dca4c.png)


- 각 은닉상태 h_j(순환 층의 유닛 개수와 길이가 동일한 벡터)가 정렬 함수을 통과하여 스칼라 값 e_J을 출력, 이 함수는 하나의 출력 유닛과 tanh 활성화 함수를 가진 단순한 완전 연결 층
- 그 다음 벡터 e_1, e_2,,, e_n에 소프트맥스 함수가 적용되어 가중치 벡터 알파1, ... 알파n을 만듦
- 마지막으로 각 은닉 상태 벡터 h_j 와 해당되는 가중치 알파j을 곱하여 더한 후 문맥 벡터 c를 만듦(따라서 c는 은닉 상태 벡터와 길이가 동일) 
<img src = "https://user-images.githubusercontent.com/72767245/105804022-1553e780-5fe2-11eb-83c8-153d2229bd70.png" width = 40%>

```python
# input이 두개임 (음표이름과, 박자에 대한 시퀀스)
# 어텐션 메커니즘은 입력의 길이가 고정될 필요 없음
notes_in = Input(shape = (None,))
durations_in = Input(shape = (None,))

# Embedding층은 음표 이름과 박자에 대한 정숫값을 벡터로 변환
x1 = Embedding(n_notes, embed_size)(notes_in) # (None, None, 100)
x2 = Embedding(n_duration, embed_size)(durations_in) # (None, None, 100)

# 이 벡터는 하나의 긴 벡터로 연결되어 순환층의 입력으로 사용
x = Concatenate()([x1,x2])  # (None, None, 200)

# 두개의 적층 LSTM 층을 사용
# 마지막 은닉상태만이 아니라 전체 은닉 상태의 시퀀스를 다음 층에 전달하기 위해 return_sequences 매개변수는 True
x = LSTM(rnn_units, return_sequences = True)(x) # (None, None, 256)
x = LSTM(rnn_units, return_sequences = True)(x) # (None, None, 256)

# 정렬 함수는 하나의 출력 유닛과 tanh 활성화 함수를 가진 단순한 Dense 층
# Reshape 층을 사용해 출력을 하나의 벡터로 펼침 
# 이 벡터의 길이는 입력 시퀀스의 길이(seq_len)과 동일
e = Dense(1, activation = 'tanh')(x)  # (None, None, 1)
# 배치 차원을 제외한 나머지 크기를 바꾸어줌
e = Reshape([-1])(e) # (None, None)

# 정렬된 값에 소프트맥스 함수를 적용하여 가중치를 계산
alpha = Activation('softmax')(e)  # (None, None)

# 은닉 상태의 가중치 합을 얻기 위해 RepeatVector층으로 이 가중치를 rnn_units 번 복사해 [rnn_units, seq_len] 크기의 행렬 만듦 
# permute : 전치 [seq_len, rnn_units]
c = Permute([2,1])(RepeatVector(rnn_units)(alpha))  # (None, 256, None) > (None, None, 256)
# 이 행렬과 마지막 LSTM 층의 은닉상태와 원소별 곱셈을 수행 , 크기: [seq_len, rnn_units]
c = Multiply()([x,c]) # [ (None, None, 256), (None, None, 256) ] > (None, None, 256)
# Lambda 층을 사용하여 seq_len축을 따라 더하여 rnn_units 길이의 문맥벡터 만듦
c = Lambda(lambda xin: K.sum(xin, axis = 1), output_shape = (rnn_units, ))(c) # (None, 256)

# 이 네트워크의 출력은 두 개, 하나는 다음 음표 이름, 하나는 다음 음표의 길이
notes_out = Dense(n_notes, activation = 'softmax', name = 'pitch')(c) # (None, 387)
durations_out = Dense(n_durations, activation = 'softmax', name = 'duration')(c) # (None, 18)

# 최종 모델은 이전 음표 이름과 박자를 입력으로 받고 다음 음표 이름과 박자에 대한 분포를 출력
model = Model([notes_in, durations_in], [notes_out, durations_out])

# 네트워크가 순환 층의 은닉 상태에 어떻게 가중치를 부여하는지 알아보기 위해 alpha 벡터를 출력하는 모델을 만듦
att_model  = Model([notes_in, durations_in], alpha)

# 음표 이름과 박자 출력은 모두 다중 분류 문제이므로 categorical_crossentropy을 사용하여 모델을 컴파일
opti= RMSprop(lr = 0.001)
model.compile(loss = ['categorical_crossentropy', 'categorical_crossentropy'], optimizer = opti)
```

### Attention을 사용한 RNN 분석
