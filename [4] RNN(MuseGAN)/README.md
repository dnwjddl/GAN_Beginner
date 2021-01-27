# 음악 창작
- 모델이 순차적인 음악의 구조를 학습하고 재생성할 수 있어야 함
- **이산적인 확률 집합**을 사용하여 **연속적인 악보**를 만들 수 있어야 함

### Attention Mechanism
RNN이 다음에 나올 음표를 예측하기 위해 **이전의 어느 음표에 초점을 맞출지** 선택 가능  
**해당 글에서는 하나의 선율을 가진 음악 생성에 초점을 맞추어 문제를 단순화**

## 준비사항
```python
from music21 import converter
original_score = converter.parse(file).chordify()
original_score.show()
original_score.show('text')
```
- ```music21``` 라이브러리 사용
  - `chordify` 메서드 : 여러 파트로 나누어진 음표를 하나의 파트에서 동시에 연주되는 화음으로 압축
- ```original_score.show('text')``` : 음악기호를 text 형식으로 볼 수 있음
  - 정보(예): ```{5.0} <music21.chord.Chord C4>```
  - 코드, pitch(A~G) : C
  - 옥타브(숫자) : 4
  - 박자(duration)(숫자) : 5번째


## RNN
- Dataset: 피치와 박자를 정숫값으로 변환
  - 피치와 박자에 대한 정수 룩업(look up) 딕셔너리
  - ```note_to_int``` & ```duration_to_int```
- batch_size: 32
- Target
  - 시퀀스에 있는 (one-hot encoding된) 다음 피치와 박자
### Attention Mechanism
#### ```AutoEncoder```
- z(문맥을 담은 벡터)/ 병목될 가능성이 있음
- 긴 문장의 시작 부분에 정보는 문맥 벡터에 도달할 때 희석될 수 있음

#### ```Attention Mechanism```
- Attention Mechanism에서는 인코더 RNN의 마지막 은닉상태만 문맥 벡터로 사용하지않고 인코더 RNN의 이전 타임 스텝에 있는 은닉상태의 가중치 합으로 문맥벡터 생성
- Attention Mechanism은 인코더의 이전 은닉상태와 현재 은닉상태를 문맥 벡터 생성을 위한 덧셈 가중치로 변환하는 일련의 층

### Attention Mechanism 생성
![image](https://user-images.githubusercontent.com/72767245/105805793-1edf4e80-5fe6-11eb-95a1-7550fc9dca4c.png)



### Attention을 사용한 RNN 분석

### Encoder-Decoder 네트워크의 어텐션
### 다중선율 음악 생성
  
