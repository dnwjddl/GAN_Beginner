# 음악창작
- 모델이 순차적인 음악의 구조를 학습하고 재생성할 수 있어야 함
- 이산적인 확률 집합을 사용하여 연속적인 악보를 만들 수 있어야 함
#### 고려해야할 요소
- 피치(pitch): 음, 높이
- 리듬(rhythm)
- 선율(polyphonic)
  - 텍스트 생성은 하나의 텍스트 스트림만 처리하면 되지만 음악은 화음의 스트림을 동시에 처리해야 함
#### Attention Mechanism
텍스트 생성에 사용한 여러가지 RNN 기술을 음악생성에서도 사용 가능(+Attention Mechanism)  
이를 사용하면 RNN 이 다음에 나올 음표를 예측하기 위해 이전의 어느 음표에 초점을 맞출지 선택 가능

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
  - 이 음표는 음악의 4번째 박자(0부터 시작)에서 시작하고
  - 다음 음표가 5번째 박자에서 시작하므로 1박자 길이를 가짐
  - 낮은 G, D, B 코드로 구성되어 있음 
- `{6.0} <music21.chord.Chord G3>` `{6.25} <music21.chord.Chord D3>`
  - 이 음표는 6번째 박자부터 시작하고 다음음표가 6.25 부터 시작하므로 1/4박자 길이를 가지고 G 코드 하나로 구성
- `{7.75} <music21.chord.Chord C4>` `{8.0} <music21.chord.Chord D4>`
  - 이 음표는 7.75번째 박자로부터 시작하고 1/4박자 길이를 가진다. 높은 C 코드 하나로 구성
