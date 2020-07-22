# 01. Torchtext를 이용한 자연어 처리

pytorch에서는 텍스트에 대한 여러 추상화 기능을 제공하는 Torchtext를 제공합니다.

토치 텍스트가 제공하는 기능들은 다음과 같습니다.

* 파일 로드하기 : 다양한 포맷의 코퍼스를 로드한다.
* 토큰화 : 문장을 단어 단위로 분리한다.
* 단어 집합 : 단어 집합을 만듭니다.
* 정수 인코딩 : 전체 코퍼스의 단어들을 고유한 정수로 맵핑한다.
* 단어 벡터 : 단어 집합의 단어들에 고유한 임베딩 벡터를 만들어 준다. 랜덤으로 초기화 된 값이 거나, 사전 훈련된 임베딩 벡터들이 로드된다.
* 배치화 : 훈련 샘플들의 배치를 만들어 준다. 이 과정에서 패딩 작업이 이루어진다.





## 1. 훈련 데이터와 테스트 데이터로 분리

```python
import urllib.request
import pandas as pd

#IMDB 데이터 다운 받기
urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")

#IMDB 데이터를 데이터프레이에 저장
df = pd.read_csv('IMDb_Reviews.csv', encoding='latin1')

#데이터 분리
train_df = df[:25000]
test_df = df[25000:]
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)
```



## 2. 필드 정의하기

```python
from torchtext import data

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=str.split,
                  lower=True,
                  batch_first=True,
                  fix_length=100)

LABEL = data.field(sequential=False,
                   use_vocab=False,
                   batch_first,False,
                   is_target=True)
```

- sequential : 시퀀스 데이터 여부. (True가 기본값)
- use_vocab : 단어 집합을 만들 것인지 여부. (True가 기본값)
- tokenize : 어떤 토큰화 함수를 사용할 것인지 지정. (string.split이 기본값)
- lower : 영어 데이터를 전부 소문자화한다. (False가 기본값)
- batch_first : 미니 배치 차원을 맨 앞으로 하여 데이터를 불러올 것인지 여부. (False가 기본값)
- is_target : 레이블 데이터 여부. (False가 기본값)
- fix_length : 최대 허용 길이. 이 길이에 맞춰서 패딩 작업(Padding)이 진행된다



## 3. 데이터셋 만들기

```python
from torchtext.data import TabularDataset

train_data, test_data = TabularDataset.splits(
    path='.', train='train_data.csv', test='test_data.csv', format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True
)

print(vars(train_data[0]))
print(train_data.fields.items())
```

- path : 파일이 위치한 경로.
- format : 데이터의 포맷.
- fields : 위에서 정의한 필드를 지정. 첫번째 원소는 데이터 셋 내에서 해당 필드를 호칭할 이름, 두번째 원소는 지정할 필드.
- skip_header : 데이터의 첫번째 줄은 무시.



## 4. 단어 집합 만들기

토큰화 전처리를 끝냈다면, 정수 인코딩 작업이 필요하다.

정수 인코딩 작업을 하려면 우선 단어 집합을 만들어주어야 합니다.

```python
TEXT.build_vocab(train_data, min_freq=10, max_size=10000)

print('단어 집합의 크기 : {}'.format(len(TEXT.vocab)))
#단어 집합의 크기 : 10002
```

- min_freq : 단어 집합에 추가 시 단어의 최소 등장 빈도 조건을 추가.
- max_size : 단어 집합의 최대 크기를 지정.

단어 집합의 크기를 10000개로 제한하였지만 실제 단어 크기는 10002개 입니다.

이는 토치텍스트가 <unk>, <pad>를 추가하였기 때문입니다.



## 5. 토치텍스트의 데이터로더

데이터로더는 데이터셋에서 미니 배치만큼 데이터를 로드하게 만들어주는 역할을 합니다. 

토치텍스트에서는 Iterator를 사용하여 데이터로더를 만듭니다.

```python
from torchtext.data import Iterator

batch_size=5
train_loader = Iterator(dataset=train_data, batch_size=batch_size)
test_loader = Iterator(dataset=test_data, batch_size=batch_size)


```



# 02. 자연어 처리에서 단어  표현

컴퓨터는 문자보다는 숫자를 더 잘 처리 할 수 있기 때문에 자연어 처리에서 정수 인코딩을 수행합니다.

* 원-핫 인코딩
* 워드 임베딩(Word Embedding)



원-핫 인코딩의 한계

* 단어의 개수가 늘어날 수록 벡터의 크기가 늘어나 저장 공간 측면에서 비효율적이다
* 단어간의 유사한 정도를 알 수  없다.



원-핫 인코딩에 대해서 잘 아신다는 가정하에 워드 임베딩을 바로 설명해 드리겠습니다.

## 01. 워드 임베딩

워드 임베딩이란 단어를 벡터로 표현하는 것을 말합니다.

워드 임베딩은 단어를 밀집 표현으로 변환하여 줍니다. 

그렇다면 밀집 표현과 워드 임베딩에 대한 개념을 이해 해봅시다.

### 1. 희소 표현

밀집 표현을 말하기 이전에 먼저 희소 표현에 대한 개념을 알아 봅시다.

원-핫 벡터들은 표현하고자 하는 단어의 인덱스의 값만 1이고 나머지는 전부 0으로 표현되는 벡터 표현 방법입니다. 이렇게 벡터나 행렬의 값이 대부분 0으로 표현되는 방법을 희소 표현이라고 합니다.



### 2. 밀집 표현

밀집 표현은 앞서 소개했던 희소 표현과는 반대되는 표현입니다.

밀집 표현은 벡터의 차원이 단어 집합의 크기로 상정되지 않고 사용자가 설정한 값으로 차원을 맞춥니다.

또한 이 과정에서 0과 1만 가진 값이 아니라 실수 값을 가지며, 희소표현처럼 대부분의 값이 0이 되지도 않습니다.

이 경우 벡터의 차원이 조밀해졌다고 하여 밀집 벡터라고 합니다,

ex) 강아지 = [0.2 1.8 1.1 -2.1 1.1 2.8 ... 중략 ...]



### 3. 워드 임베딩(Word Embedding)

그래서 결국 워드 임베딩을 한다는 것은 단어를 원-핫 인코딩(희소 표현)에서 워드 임베딩(밀집 표현)으로 변환 한다는 것이라고 할 수 있습니다.

밀집 벡터는 워드 임베딩 과정을 통해 나온 결과라고 하여 임베딩 벡터라고도 합니다.

워드 임베딩의 가장 특이한 점은 단어 벡터를 인공 신경망을 통해 학습할 수 있다는 점입니다.

| -         | 원-핫 벡터               | 임베딩 벡터              |
| --------- | ------------------------ | ------------------------ |
| 차원      | 고차원(단어 집합의 크기) | 저차원                   |
| 다른 표현 | 희소 벡터의 일종         | 밀집 벡터의 일종         |
| 표현 방법 | 수동                     | 훈련 데이터로부터 학습함 |
| 값의 타입 | 1과 0                    | 실수                     |



# 03. pytorch nn.Embedding()

* 처음부터 임베딩 벡터를 학습하는 방법

* 사전에 훈련된 임베딩 벡터들을 가져오는 방법

  * 내가 훈련한 임베딩 벡터

  * 외부에서 사전 훈련된 임베딩 벡터



## 1. 처음부터 하는 워드 임베딩

임베딩 층의 입력으로 사용하기 위해선 입력 시퀀스의 각 단어들은 모두 정수 인코딩이 되어있어야 합니다.



어떤 단어 --> 단어에 부여된 고유한 정수값 --> 임베딩 층 통과 --> 밀집 벡터

이 과정은 특정 단어에 부여된 정수를 인덱스로 가지는 테이블로부터 임베딩 벡터 값을 가져오는 룩업 테이블이라고 볼 수 있습니다.

그리고 이 테이블은 단어 집합의 크기만큼의 행을 가지므로 모든 단어는 고유한 임베딩 벡터를 가집니다.

이제 nn.Embedding()으로 사용할 경우를 봅시다.

```python
import torch.nn as nn
train_data = 'you need to know how to code'
word_set = set(train_data.split()) # 중복을 제거한 단어들의 집합인 단어 집합 생성.
vocab = {tkn: i+2 for i, tkn in enumerate(word_set)}  # 단어 집합의 각 단어에 고유한 정수 맵핑.
vocab['<unk>'] = 0
vocab['<pad>'] = 1

embedding_layer = nn.Embedding(num_embeddings=len(vocab),
                               embedding_dim=3,
                               padding_idx=1)

#num_embeddings : 임베딩을 할 단어들의 개수
#embedding_dim : 임베딩 할 벡터의 차원
#padding_idx : 패딩을 위한 토큰의 인덱스

print(embedding_layer.weight)
```

여기 까진 임베딩 벡터를 초기화한 상태로 생성만 했을 뿐입니다. 손실함수와 옵티마이저를 이용하여 임베딩 벡터를 학습하는 방법에 대해서는 나중에 다루도록 하겠습니다.



## 2. 사전 훈련된 워드 임베딩

사전 훈련된 워드 임베딩은 두가지 방법이 잇습니다

첫번째는 내가 훈련한 임베딩 벡터를 가져오는 방법, 두번째는 외부에서 가져온 사전 훈련된 임베딩 벡터를 사용하는 방법입니다. 

사전 훈련된 임베딩 벡터를 들고오기전에 가장 먼저 임베딩 벡터에 맵핑시킬 단어받아와야 합니다.

```python
from torchtext import data, datasets

TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=True, batch_first=True)

trainset, testset = datasets.IMDB.splits(TEXT, LABEL)
```



* 토치텍스트의 Field 객체의 build_vocab을 사용하여 사전 훈련된 워드 임베딩 벡터를 사용할 수 있습니다.

### 1. 내가 훈련한 임베딩 벡터를 가져오는 방법

```python
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
from torchtext.vocab import Vectors
word2vec_model = KeyedVectors.load_word2vec_format('eng_w2v')

print(word2vec_model['this'])

vectors = Vectors(name='eng_w2v') #사전 훈련된 Word2Vec 모델의 임베딩 벡터를 vectors에 저장
TEXT.build_vocab(trainsets, vectors=vectors,max_size=10000, min_freq=10) #다시 한번 강조하면 Field 객체의 build_vocab을 사용하여 사전 훈련된 워드 임베딩을 사용할 수 있습니다!

print(TEXT.vocab.stoi) #현재 단어 집합의 단어와 맵핑된 정수 확인
print(TEXT.vocab.vectors) # 사전 훈련된 임베딩 벡터값

embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=False)#임베딩 벡터들을 nn.Embedding()을 이용해 임베딩 레이어를 초기화
```



### 2. 외부에서 가져온 사전 훈련된 워드 임베딩

토치텍스트에서는 사전 훈련된 임베딩 벡터를 제공합니다. 다음은 제공되는 임베딩 벡터 리스트의 일부입니다.

- fasttext.en.300d
- glove.42B.300d
- glove.twitter.27B.200d
- glove.6B.50d
- glove.6B.300d <= 이걸 사용해볼 겁니다.



이제 토치텍스트가 제공하는 사전 훈련된 임베딩 벡터들로 임베딩 레이어를 초기화 해봅시다.

```python
from torchtext.vocab import GloVe

TEXT.build_vocab(trainset, vectors=GloVe(name'6B', dim=300), max_size=10000, min_freq=10)
LABEL.build_vocab(trainset)

print(TEXT.vocab.vectors) #사전 훈련된 임베딩 벡터값 출력
embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=False)#임베딩 벡터들을 nn.Embedding()을 이용해 임베딩 레이어 초기화
```



