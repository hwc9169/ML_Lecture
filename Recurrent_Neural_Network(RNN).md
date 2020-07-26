# 순환 신경망(Recurrent Neural Network)

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

* 입력 값 : It
* 입력층 가중치 : Wx

* 은닉 상태값 : ht
* 출력 값 : Ot
* 출력층 가중치 : Wy
* 이전 시점 은닉 상태값 : ht-1
* 이전 시점 은닉 상태값(ht-1)의 가중치 : Wh

ht = tanh(WxIt + ht-1Wh + b)

Ot = f(Wyht + b)  단, f는 비선형 활성화 함수



* 단어 벡터의 차원  :  d
* 은닉 상태 크기 : Dh

It = (d x 1)

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

# 단어 정리

* 피드 포워드 신경망(Feed Forward Neural Network) : 모든 입력 값이 전부 은닉층에서 활성화 함수를 지나 출력층 방향으로만 향하는 신경망 예로는 ANN과 CNN이 있다.

* 셀(cell) : RNN에서 은닉층에서 활성화 함수를 거처 결과를 내보내는 노드. 이 셀은 이전의 값을 기억하려고 하는 메모리 역할도 수행하기 떄문에 메모리 셀 또는 RNN 셀이라고도 표현합니다.
* 은닉 상태값(hidden state) : 메모리 셀이 출력층 방향과 다음 노드에 보내는 값



