# 01. 순환 신경망(Recurrent Neural Network, RNN)

RNN에 대한 이해와 RNN의 장기 의존성 문제를 보완한 LSTM에 대하여 알아 보겠습니다.

RNN은 입력과 출력을 시퀀스 단위로 처리하는 시퀀스(Sequence) 모델입니다. 번역기의 경우 입력은 변역하고자 하는 문장 즉, 단어 시퀀스입니다. 번역기의 출력 값(번역된 문장) 또한 단어 시퀀스 입니다. 이러한 시퀀스를 처리하기 위해서 고안된 모델이 시퀀스(Sequence)모델이라고 합니다. 

그래서 우리가 배우려 하는 이 RNN은 딥 러닝의 가장 기본적인 시퀀스 모델이라고 할 수 있습니다.



## 1. 순환 신경망

### 기본 개념

보통 우리가 알고 있는 CNN이나 ANN과 같은 신경망들은 모든 값들은 전부 은닉층에서 활성화 함수를 지나 출력층 방향으로만 향했습니다. 이와 같은 신경망들을 피드 포워드 신경망(Feed Forward Neural Network)이라고 합니다. 하지만 순환 신경망(RNN)은 위와 같지 않은 신경망 중에 하나 입니다.  RNN은 은닉층의 노드에서 활성화 함수를 통해 나온 결과값을 출력층 방향에 출력할 뿐만 아니라 다음 계산의 입력으로 보내 집니다.

*  CNN, ANN은 모든 입력값들이 출력층 방향으로만 향한다.
* RNN은 입력값들이 두 방향, 출력층 방향과 다음 계산의 입력으로 들어간다.



피드 포워드 신경망에서는 뉴런이라는 단위를 사용했지만, RNN에서는 입력층의 뉴런을 입력 벡터, 출력층의 뉴런을 출력 벡터, 은닉층의 뉴런은 은닉 상태라는 표현을 사용합니다. 

RNN은 입력과 출력의 길이를 다르게 설계 할 수 있기 때문에 다양한 용도로 사용됩니다. RNN 셀의 각 입출력 단위는 가장 보편적으로 '단어 벡터'입니다. 

예를 들어하나의 이미지 입력에 대해서 사진의 제목을 출력하는(one-to-many) 이미지 캡셔닝 작업에 사용할 수 있습니다. 사진의 제목은 문장 즉, 시퀀스가 출력 됩니다.

또한 단어 시퀀스(문장) 에 대해서 하나의 출력(many-to-one)을 하는 모델은 입력 문서가 긍정적인지 부정적인지 판별하는 감성 분류(sentiment classification), 또는 메일이 정상 메일인지 스팸 메일인지 판별하는 스팸 메일 분류(spam detection)에 사용 됩니다.

다 대 다 (many-to-many)의 모델은 입력 문장으로 부터 대답 문장을 출력하는 챗봇이나 변역기, 품사 태깅과 같은 작업이 가능합니다.



### RNN에 대한 수식

* 입력 값 : xt

* 입력층 가중치 : Wx

* 은닉 상태값 : ht

* 출력 값 : yt

* 출력층 가중치 : Wy

* 이전 시점 은닉 상태값 : ht-1

* 이전 시점 은닉 상태값(ht-1)의 가중치 : Wh

  ht = tanh(Wx xt + Wh ht-1 + b)

  yt = f(Wyht + b)  단, f는 비선형 활성화 함수



* 단어 벡터의 차원  :  d

* 은닉 상태 크기 : Dh

  xt = (d x 1)

  Wx = (Dh x d)

  Wh = (Dh x Dh)

  ht-1 = (Dh x 1)

  b = (Dh x 1)



## 2. 파이썬으로 RNN 구현하기

```python
import numpy as np

timesteps = 10
input_size = 4
hidden_size = 8

inputs = np.random.random((timesteps,input_size))

hidden_state_t = np.zeros((hidden_size,))

Wx = np.random.random((hidden_size,input_size))
Wh = np.random.random((hidden_size,hidden_size))
b = np.random.random((hidden_size,))

total_hidden_states = []

for input_t in inputs:
    output_t = np.tanh(np.dot(Wx,input_t) + np.dot(Wh,hidden_state_t) + b)

    total_hidden_states.append(list(output_t))
    print(np.shape(total_hidden_states))

    hidden_state_t = output_t

total_hidden_states = np.stack(total_hidden_states)
print(total_hidden_states)
```

* timesteps는 시점의 수를 말하며 NLP에서는 문장의 단어의 갯수를 의미한다.
* input_size는 NLP에서 각 단어의 임베딩 벡터 차원을 의미한다.
* hidden_size는 은닉 상태의 크기를 말한다. 메모리 셀의 용량.



## 3. 파이토치로 RNN 구현하기

```python
import torch
import torch.nn as nn

input_size = 5
hidden_size = 8

inputs = torch.Tensor(1, 10, 5)

cell = nn.RNN(input_size, hidden_size, batch_first=True)
outputs, status = cell(inputs)

print(outputs.shape) #torch.Size([1, 10, 8])
print(status.shape) #torch.Size([1, 1, 8])
```

nn.RNN의 인자 값은 차례대로 입력의 크기, 은닉 상태의 크기 그리고 마지막 batch_first=True를 통해 입력  텐서의 첫번째 차원이 배치 크기임을 알려줍니다

RNN 셀은 두개의 입력을 리턴합니다 첫번째 리턴값은 모든 시점의 은닉 상태값, 두번째 리턴값은 마지막 시점의 은닉 상태값 입니다.



## 4. 깊은 순환 신경망(Deep Recurrent Neural Network)

RNN도 다수의 은닉층을 가질 수 있습니다. 첫번째 은닉층이 다음 은닉층의 모든 시점에 대해서 은닉 상태값을 다음 은닉층으로 보내주면 됩니다.

깊은 순환 신경망을 파이토치로 구현할 때는 nn.RNN()의 인자로 num_layers에 값을 전달하여 층을 쌓으면 됩니다. 층이 2개인 깊은 신경망의 경우를 살펴보겠습니다.

```python
import torch.nn as nn
import torch

inputs = torch.Tensor(1, 10, 5)
cell = nn.RNN(input_size=5, hidden_size=8, num_layers=2, batch_first=True)
outputs, status = cell(inputs)

print(outputs.shape) #torch.Size([1, 10, 8])
print(status.shape) #torch.Size([2, 1, 8])
```

첫번째 리턴값의 크기는 층이 1개였던 RNN 셀 때와 같습니다. 여기서는 두번째 층의 모든 시점의 은닉 상태들입니다.

두번째 리턴값의 크기는 층이 1개였던 RNN셀 때와 달리 2입니다. 여기서 크기는 (층의 개수, 배치 크기, 은닉 상태의 크기)에 해당합니다,


## 5. 양방향 순환 신경망(Bidirectional Recurrent Neural Network)

양방향 순환 신경망은 시점 t에서의 출력값을 예측할 때 이전 시점의 데이터뿐만 아니라, 이후 데이터로도 예측할 수 있다는 아이디어에 기반합니다. 예를 들어 하나의 문장에서 중간 단어를 예측할 때 앞 단어 뿐만 아니라 뒷 단어도 고려한다면 쉽게 예측할 수 있다는 말입니다.

즉, RNN이 과거 시점(time step)의 데이터들을 참고해서, 정답을 예측하지만 실제 문제에서는 과거 시점의 데이터뿐만 아니라 향후 시점의 데이터도 고려해야 합니다. 그래서 이전 시점과 이후 시점의 데이터를 가지로 예측을 하기 위해 고안된 것이 양방향 RNN입니다.

양방향 RNN은 하나의 출력값을 예측하기 위해 기본적으로 두개의 메모리 셀을 사용합니다. 첫 번째 메모리 셀은 앞 시점의 은닉 상태를 전달받아 현재의 은닉 상태를 계산하고(일반적인 RNN과 같은 동작), 두 번째 메모리 셀은 앞 시점의 은닉 상태가 아니라 뒤 시점의 은닉 상태를 전달 받아 현재의 은닉 상태를 계산합니다. 그리고 이 두 메모리 셀을 사용하여 출력값을 예측합니다.

물론, 양방향 RNN도 다수의 은닉층을 가질 수 있습니다. 다른 인공 신경망 모델들도 마찬가지지만, 은닉층을 추가한다고 해서 모델의 성능이 무조건 좋아지는 것이 아닙니다. 학습할 수 있는 양이 ㅁ낳아지지만 반대로 훈련 데이터가 그만큼 많이 필요합니다.

양방향 순환 신경망을 파이토치로 구현할 때는 nn.RNN()의 인자로 bidirectional에 값을 True로 전달하면 됩니다.

```python
import torch.nn as nn
import torch

inputs = torch.Tensor(1, 10, 5)
cell = nn.RNN(input_size=5, hidden_size=8, num_layers=2, batch_first=True, bidirectional=True)
outputs, status = cell(inputs)

print(outputs.shape) #torch.Size([1, 10, 8x2])
print(status.shape) #torch.Size([2x2, 1 ,8])
```

첫번째 리턴값의 크기는 단방향 RNN 셀 때보다 은닉 상태의 크기가 두 배가 됩니다. (배치 크기, 시퀀스 크기, 은닉 상태의 크기x2)

두번째 리턴값의 크기는 (층의 개수x2, 배치 크기, 은닉 상태의 크기)를 가집니다. 이는 정방향 기준으로는 마지막 시점이 면서 역방향 기준에서는 첫번째 시점에 해당되는 출력값을 층의 개수만큼 쌓아 올린 결과 입니다.

# 02. 장단기 메모리(Long Short-Term Memory, LSTM)

LSTM이란 RNN의 한계를 극복하기 위한 다양한 RNN의 변형 중 하나입니다. 앞으로의 설명에선 LSTM과 RNN을 비교하여 설명하겠습니다.



## 1. RNN의 한계

앞에서 보았듯이 RNN은 출력 결과가 이전 계산 결과에 종속적입니다. 하지만 일반적인 RNN(이하 바닐라 RNN)은 짧은 시퀀스(sequence)에 대해서만 효과를 보이는 단점이 있습니다. 바닐라 RNN의 시점(time step)이 길어질 수록 앞의 정보가 뒤로 충분히 전달되지 못하는 현상이 발생합니다. 뒤로 갈수록 처음의 입력의 정보량이 손실됩니다. 

어쩌면 가장 중요한 정보가  쪽에 위치할 수 있습니다. 예를 들어 ''모스크바에 여행을 왔는데 건물도 예쁘고 먹을 것도 맛있었어. 그런데 글쎄 직장 상사한테 전화가 왔어. 어디냐고 묻더라구 그래서 나는 말했지. 저 여행왔는데요. 여기 ___'' 다음 단어를 예측하기 위해서는 맨 앞에 위치한 '모스크바'를 RNN이 충분히 기억하지 못한다면 다음 단어를 엉뚱하게 예측합니다.

이를 장기 의존성 문제(the problem of Long-Term Dependencies)라고 합니다.



## 2. 바닐라 RNN의 내부

LSTM에 대해 이해하기 전에 바닐라 RNN의 뚜껑을 열어보겠습니다. 

ht : 현재 시점의 은닉 상태값

ht-1 : 이전 시점의 은닉 상태값

Wx : 입력 가중치

Wh : 은닉 상태 가중치

xt : 현재 시점의 입력값

yt : 현재 시점의 출력값



ht = tanh(Wx xt + Wh ht-1 + b)



바닐라 RNN은 it와 ht-1이라는 두개의 입력이 각각의 가중치와 곱해져서 메모리 셀의 값이 됩니다. 그리고 이를 하이퍼볼릭탄젠트(tanh) 함수의 입력으로 사용하고 이 값은 은닉층의 출력인 은닉 상태가 됩니다.



## 3. LSTM(Long Short-Term Memory)

바닐라 RNN의 "시점이 충분히 길 때 앞의 정보가 손실된다."는 단점을 보완한 RNN의 일종을 장단기 메모리(Long Short-Term Memory)라고 하며, 줄여서 LSTM이라고 합니다. LSTM은 은닉층에 메모리 셀에 입력 게이트, 삭제 게이트, 출력 게이트를 추가하여 불필요한 기억을 지우고, 필요한 기억들을 정합니다. 쉽게 말해서 LSTM은 은닉 상태를 계산하기가 복잡하고 셀 상태(cell state)라는 값을 추가하였습니다. 



셀 상태 또한 은닉 상태와 같이 이전 시점의 값이 현재 시점의 셀 상태의 입력으로 사용됩니다. 은닉 상태값과 셀 상태값을 구하기 위해서 새로 추가 된 3개의 게이트를 사용합니다. 각 게이트는 삭제 게이트, 입력 게이트, 출력 게이트 이며 이 3개의 게이트 모두 공통적으로 시그모이드 함수가 존재합니다. 시그모이드 함수를 지나면 0과 1 사이의 값이 나오게 되는데 이 값들을 가지고 게이트를 조절합니다. 아래의 내용과 함께 각 게이트에 대해서 알아보도록 하겠습니다.



* 이하 식에서 sigmoid는 시그모이드 함수를 의미합니다.
* 이하 식에서 tanh는 하이퍼볼릭탄젠트 함수를 의미합니다.
* Wxi, Wxg, Wxf, Wxo는 각각 it와 함께 게이트에서 사용되는 4개의 가중치입니다.
* Whi, Whg, Whf, Who는 각각 ht-1와 함께 게이트에서 사용되는 4개의 가중치입니다.
* bi, bg, bf, bo는 각 게이트에서 사용되는 4개의 편향입니다.



### (1) 입력 게이트

 입력게이트는 현재 정보를 기억하기 위한 게이트입니다. 우선 현재 시점 t의 x값과(input 값) 입력게이트로 이어지는 가중치 Wxi를 곱하고 이전시점 t-1의 ht-1(은닉 상태)가 입력게이트로 이어지는 Whi를 곱한 값을 더하여 시그모이드 함수를 지납니다. 이를 it라고 합니다.

it = sigmoid(Wxi xt + Whi ht-1 + bi)



그리고 현재 시점 t의  x값과 입력 게이트로 이어지는 가중치 Wxi를 곱한 값과 이전 시점 t-1의  ht-1가 입력 게이트로 이어지는 가중치 wxi를 곱한값을 더하여 하이퍼볼릭탄젠트 함수를 지납니다. 이를 gt라고 합니다

gt = tanh(Wxi xt + Whi ht-1 + bi)



시그모이드 함수를 지나면 0과 1 사이의 값이 하이퍼볼릭탄젠트 함수를 지나면 -1과 1 사이의 값 두 개가 나오게 됩니다. 이 두개의 값을 가지고 이번에 선택된 기억할 정보의 양을 정하는데, 구체적으로 어떻게 결정하는지 알아보겠습니다.



### (2) 삭제 게이트

삭제 게이트는 기억을 삭제하기 위한 게이트입니다. 현재 시점 t의 x값과 이전 시점 t-1의 ht-1이 시그모이드 함수를 지나게 됩니다. 시그모이드 함수를 지나면 0과 1 사이의 값이 나오게 되는데, 이 값이 곧 삭제 과정을 거친 정보의 양입니다. 0에 가까울수록 정보가 많이 삭제된 것이고 1에 가까울수록 온전히 기억된 것입니다. 이를 가지고 셀 상태를 구하게 됩니다.

ft = sigmoid(Wxfxt+Whfht−1+bf)



### (3) 셀 상태(장기 상태)

셀 상태(Ct)를 LSTM에서 장기 상태라고 부릅니다. 그렇다면 셀 상태를 구하는 방법을 알아 보겠습니다. 삭제 게이트에서 일부 기억을 잃은 상태입니다. 

입력 게이트에서 구한 it, gt 두 개의 값에 대해서 원소별 곱(entrywise product)을 진행합니다. (여기서 ∘가 연산자로 사용되었습니다. ) 다시 말해 같은 크기의 두 행렬이 있을 때 같은 위치의 성분끼리 곱하는 것을 말합니다.  

입력 게이트에서 선택된 기억을 삭제 게이트의 결과값과 원소별 곱을 하고, 이 값을 입력 게이트의 계산 결과와 덧셈을 합니다. 이를 현재 시점 t의 셀 상태(Ct)라고 하며, 이 값은 다음 시점 t+1의 LSTM셀로 넘겨집니다.

Ct = ft ∘ Ct-1 + it ∘ gt



그렇다면 삭제 게이트와 입력 게이트의 영향력을 이해해봅시다. 삭제 게이트 ft가 0이 된다면, 이전 시점의 셀 상태값 Ct-1이 현재 시점의 셀 상태값 Ct에게 아무런 영향을 끼치지 않게됩니다. 이는 삭제 게이트가 완전히 닫히고 입력 게이트를 연 상태를 의미합니다. 반댈 입력 게이트의 it 값을 0이라고 한다면, 현재 시점의 셀 상태 값 Ct는 오직 이전 시점의 Ct-1의 영향만을 받습니다. 이는 입력 게이트를 완전히 닫고 삭제 게이트만을 연 상태를 의미합니다. 결론을 정리하면 삭제 게이트는 이전 시점의 입력을 얼마나 반영할지 결정하고, 입력 게이트는 현재 시점의 입력을 얼마나 반영할지 결정합니다.



### (4) 출력 게이트와 은닉 상태(단기 상태)

출력 게이트는 현재 시점 t의 xt값과 이전 시점 t-1의 은닉 상태 ht-1 시그모이드 함수를 지난 값입니다.  출력 게이트는 현재 시점 t의 은닉 상태 ht를 결정합니다

ot = sigmoid(Wxo xt + Who ht-1 + bo)



은닉 상태 ht를 단기 상태라고 부르는데, 은닉 상태는 장기 상태의 값이 tanh 함수를 지나 -1과 1사이의 값이 됩니다. 해당 값은 출력 게이트의 값과 연산되면서, 값이 걸러지는 효과가 발생합니다. 단기 상태의 값은 출력층으로도 향합니다.

ht =  ot ∘ tanh(Ct)



## 4. 파이토치의 nn.LSTM()

파이토치에서 LSTM 셀을 사용하는 방법은 매우 간단합니다. 기존 RNN 셀을 사용하려 했을 때와 비교하며 설명해 보겠습니다.

```python
#기존의 RNN 코드
nn.RNN(input_dim, hidden_size, batch_first=True)

#LSTM 코드
nn.LSTM(input_dim, hidden_size, batch_first=True)
```



# 03. 다대다 RNN

이번에는 모든 시점의 입력에 대해서 모든 시점에 대해서 출력을 하는 다대다 RNN을 구현해봅시다.

다대다 RNN의 대표적인 사용법

* 품사 태깅
* 개체명 인식



## 1. 문자 단위 RNN

입출력의 단위가 단어가 아니라 문자(char) 레벨로 하여 RNN을 구현합니다. RNN 구조가 달라진 것이 아니고 입출력 단위가 문자로 바뀌었을 뿐입니다. 그렇다면 다대다 문자 단위 RNN을 구현해봅시다.

우선 필요한 도구들을 임포트합니다.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
```



### 1. 훈련 데이터 전처리

여기선 문자 시퀀스 apple을 받아 pple!를 출력하는 RNN을 구현해보겠습니다. 이렇게 구현하는 어떤 의미가 있는 것은 아니고, 그저 RNN의 동작을 이해하기 위한 목적입니다.

입력 데이터와 레이블 데이터에 대해서 문자 집합(vocabulary)을 만듭니다.

```python
input_str = 'apple'
label_str = 'pple!'
char_vocab = sorted(list(set(input_str+label_str)))
vocab_size = len(char_vocab)
print('문자 집합 크기 : {}'.format(vocab_size)) #문자 집합 크기 : 5
```

현재 문자 집합에 총 5개의 문자 ' ! ', ' a ', ' e ', ' l ', ' p '가 있습니다. 이제 하이퍼파라미터를 정의 해보겠습니다.

```python
input_size = vocab_size #입력의 크기는 문자 집합 크기
hidden_size = 5
output_size = 5
learning_rate = 0.1
```

이제 문자 집합에 고유한 정수를 부여합니다.

```python
char_to_index = { v : i for i, v in enumerate(char_vocab)}
print(char_to_index) #{'!': 0, 'a': 1, 'e': 2, 'l': 3, 'p': 4}
```

반대로 정수를 통해 문자를 얻을 수 있는 index_to_char을 만듭니다.

```python
index_to_char = { i : v for i, v in enumerate(char_vocab)}
print(index_to_char) #{0: '!', 1: 'a', 2: 'e', 3: 'l', 4: 'p'}
```

이제 입력 데이터와 레이블 데이터의  각 문자에 vocabulary을 이용하여 정수로 맵핑해보겠습니다.

```python
x_data = [index_to_char[i] for i in list(input_str)]
y_data = [char_to_index[i] for i in list(label_str)]
print(x_data) #[1, 4, 4, 3, 2]
print(y_data) #[4, 4, 3, 2, 0]
```

파이토치의 nn.RNN()은 기본적으로 3차원 텐서를 입력받습니다.  

```python
 x_data	= x_data.unsqueeze(0)
 y_data = y_data.unsqueeze(0)
print(x_data) #[[1, 4, 4, 3, 2]]
print(y_data) #[[4, 4, 3, 2, 0]]
```

입력 시퀀스의 각 문자들을 원-핫 벡터로 바꿔줍니다

```python
x_one_hot = [np.eye(vocab-size)[x] for x in x_data]
print(x_one_hot)
#[array([[0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 1.],
       [0., 0., 0., 1., 0.],
       [0., 0., 1., 0., 0.]])]
```

입력 데이터와 레이블 데이터를 텐서로 바꿔줍니다.

```python
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)
print('훈련 데이터의 크기 : {}'.format(X.shape)) #훈련 데이터의 크기 : torch.Size([1, 5, 5])
print('레이블의 크기 : {}'.format(Y.shape)) #레이블의 크기 : torch.Size([1, 5])
```



### 2. 모델 구현하기

이제 RNN 모델을 구현해봅시다. 아래에서 fc는 완전 연결층(fully-conneted layer)을 의미하며 출력층으로 사용됩니다.

```python
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True) #RNN 셀 구현
        self.fc = nn.Linear(hidden_size, output_size, bias=True)
        
    def forward(self, x): # 구현한 RNN 셀과 출력층 연결
        x, status = self.rnn(x)
        x = self.fx(x)
        return x
```

이제 모델에 입력을 넣어 출력값을 확인 해봅시다.

```python
net = Net(input_size, hidden_size, output_size)
outputs = net(X)
print(outputs.shape) #torch.Size([1, 5, 5])
```

(1, 5, 5)의 크기를 가지는데 각각 배치 차원, 시점 개수, 출력의 크기입니다. 나중에 정확도를 측정할 때 이를 view를 이용하여 펼쳐서 계산합니다. 앞에서 보았듯이 레이블 데이터는 (1, 5)의 크기를 가지는데 정확도를 측정할 떄, 이걸 펼쳐서 계산할 예정입니다.

```python
print(outputs,view(-1,output_size).shape) #torch.Size([5, 5])
print(Y.view(-1).shape) #torch.Size([5])
```

이제 옵티마이저와 손실 함수를 정의해보겠습니다.

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
```

총 100번의 에포크를 학습합니다.

```python
epochs = 100

for i in range(epochs):
   	outputs = net(X)
    loss = criterion(outputs.view(-1,output_size),Y.view(-1)) #view를 통해 Batch 차원을 제거
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    result = outputs.data.numpy().argmax(axis=2) #최종 예측값인 각 time-step 별 5차워너 벡터에 대해서 가장 높은 인덱스를 선택
    result_str = ''.join([index_to_char[c] for c in np.squeeze(result)])
    
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str", result_str)
```



# 04. Pytorch를 이용한 텍스트 분류

텍스트 분류 작업의 구성

* 훈련데이터 이해
* 훈련데이터와 테스트 데이터
* 단어에 인덱스 부여
* RNN으로 분류하기



## 1. 훈련데이터 이해

텍스트 분류 작업은 지도 학습(Supervised Learning)에 속합니다. 예를 들어 스팸 메일인지 아닌지를 구분하는 경우 훈련 데이터같은 경우에는 해당 메일과 그 메일이 정상인지 스팸인지 레이블(Label)로 나타냅니다.

| 메일 (메일 내용)                 | 레이블(스팸 여부) |
| -------------------------------- | ----------------- |
| 당신에게 드리는 마지막 혜택! ... | 스팸 메일         |
| 내일 뵐 수 있을지 확인 부탁...   | 정상 메일         |
| 쉿! 혼자 보세요...               | 스팸 메일         |
| 언제까지 답장 가능할...          | 정상 메일         |
| ...                              | ...               |
| (광고) 멋있어질 수 있는...       | 스팸 메일         |

우리는 총 20, 000개의 데이터를 가지고있고, 이 데이터를 기계가 학습하게 되는데, 만약 잘 학습된 모델이라면 훈련 데이터에 없는 새로운 메일 내용을 보고 레이블을 에측할 수 있습니다. 



## 2. 훈련 데이터와 테스트 데이터

우리는 이 20,000개의 데이터로 훈련용과 테스트용으로 나누어야합니다. 예를 들어 20,000개의 샘플중에서 18,000개의 샘플은 훈련용으로 사용하고, 2000개의 샘플은 테스트용으로 보류한 채 훈련을 시킬 떄는 사용하지 않을 수 있습니다. 그리고 18,000개 데이터로 훈련한 모델에 테스트용 샘플의 레이블을 보여주지않고 맞춰보라고 요구한뒤 정확도를 확인해볼 수 있습니다. 즉, 정답률을 계산하게 됩니다.



## 3. 단어에 인덱스 부여

워드 임베딩을 이용해 밀집 벡터(dense vector)로 바꿀 수 잇습니다.Natural_Language_Processing(NLP)수업에서 보앗듯이 nn.Embedding()은 "단어 각각에 대해 정수가 맵핑된 입력"(word2index)에 대해서 임베딩 작업을 수행합니다.

단어 각각에 정수를 맵핑하는 방법은 단어의 빈도수 순대로 정렬하고 빈도수가 많은 순으로 인덱스를 부여하는 방법이 있습니다. 빈도수가 많은 순대로 인덱스를 부여하는 방법의 장점은 빈도수가 적은 단어를 제거할 수 있다는 점입니다.



## 4. RNN으로 분류하기

```python
# RNN 은닉층을 추가하는 코드
nn.RNN(input_size,hidden_size, batch_first=True)
```

RNN 코드의 timesteps와 input_dim, hidden_size를 해석해보면 다음과 같습니다.(RNN의 변형인 LSTM이나 GRU도 동일합니다.)



* hidden_size : 출력 크기(output_dim)
* timesteps : 각 문서의 단어 수
* input_size : 단어 벡터의 차원 수



## 5. RNN의 다대일(Many-to-One) 문제

텍스트 분류는 RNN의 다대일 문제입니다. 즉, 텍스트 분류의 모든 시점에 대해서 입력을 받지만 최종 시점의 RNN셀만이 은닉 상태를 출력하고, 활성화 함수를 통해 정답을 고르는 문제가 됩니다.

이 떄 선택지가 두 개뿐인 경우 이진 분류(Binary Classification) 문제가 되고, 선택지가 여러 개인경우 다중 클래스 분류(Multi-Class Classification) 문제라고 합니다. 각각 문제에 맞는 활성화 함수와 손실 함수를 사용할 것입니다.

이진 분류의 경우 활성화 함수로 시그모이드 함수를, 다중 클래스 문제의 경우 활성화 함수를 소프트맥스 함수를 사용합니다. 또한 다중 클래스 분류 문제의 경우에는 클래스가 N개라면 출력층에 해당되는 밀집층의 크기는 N으로 합니다. 즉 출력층의 뉴런수는 N개입니다. (소프트맥스 함수로 이진 분류를 할 수도 있습니다. 출력층의 뉴런을 2개로 배치하면 됩니다.)

# 단어 정리

* 피드 포워드 신경망(Feed Forward Neural Network) : 모든 입력 값이 전부 은닉층에서 활성화 함수를 지나 출력층 방향으로만 향하는 신경망 예로는 ANN과 CNN이 있다.
* 셀(cell) : RNN에서 은닉층에서 활성화 함수를 거처 결과를 내보내는 노드. 이 셀은 이전의 값을 기억하려고 하는 메모리 역할도 수행하기 떄문에 메모리 셀 또는 RNN 셀이라고도 표현
* 은닉 상태값(hidden state) : 메모리 셀이 출력층 방향과 다음 노드에 보내는 값
* 입력 게이트(gt, it) : 현재 정보를 기억하는 게이트 gt와 it가 존재하는데 it는 gt의 Ct에 대한 영향력을 결정,     수식은 it = sigmoid(Wxi xt + Whi ht-1 + bi), gt = tanh(Wxi xt + Whi ht-1 + bi)
* 삭제 게이트(ft) :  이전 시점 t의 셀 상태 Ct-1가 Ct에 대한 영향력을 결정, 수식은                                                  ft = sigmoid(Wxfxt+Whfht−1+bf)
* 출력 게이트(ot) : 은닉 상태 ht를 결정하는 게이트 수식은 ot = sigmoid(Wxo xt + Who ht-1 + bo)
* 셀 상태 (Ct) : 셀 상태, 장기 상태 모두 같은 의미이며  셀 상태 수식은 Ct = ft ∘ Ct-1 + it ∘ gt 입니다.
* 은닉 상태(ht) : 은닉상태 , 단기 상태 모두 같은 의미이며 은닉  상태는 출력 게이트의 결과값과 셀 상태값과의 계산을 통해 구할수 있다. 수식은 ht=ot∘tanh(ct)





