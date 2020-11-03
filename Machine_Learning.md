# 머신 러닝 

설명하기 앞서 이 글은 이론적인 공부들을 정리한 것 입니다. 자세한 코드 구현은 충분히 구글링으로 해결할 수 있으리라 생각합니다.

## |목차
- 머신 러닝 모델 평가
- 데이터 전처리 및 특성 추출
- 과대적합과 과소적합



## |머신 러닝 모델 평가

머신 러닝은 일반적으로 학습 데이터셋에서는 잘 동작하지만, 검증 데이터셋이나 테스트 데이터엣에는 잘 동작하지않는 문제가 존재합니다. 이런 현상을 "overfitting" 즉, 너무 과하게 학습 데이터셋에 적합하게 됩니다. 인간으로치면 흰고양이만 보고 자란 아기는 모든 고양이를 흰색이라고 생각하게 되는 것과 마찬가지입니다. 
반대로 학습 데이터셋에서 잘 동작하지 않는 현상은 과소적합(Underfitting)이라고 합니다. 

### 1. 학습, 검증, 테스트 분할

데이터는 학습 데이터셋, 검증 데이터셋, 그리고 테스트 데이터셋의 세 부분으로 나누는 방식을 많이 사용합니다. 
이를 홀드아웃 데이터셋(holdout dataset)이라고 합니다. 바로 일반적인 홀드아웃 데이터셋의 방법을 알아 보겠습니다.


1. 학습 데이터셋으로 알고리즘 내부의 가중치 학습
2. 검증 데이터셋으로 알고리즘의 하이퍼파라미터 튜닝
3. 1 ~ 2을 반복 수행
4. 테스트 데이터셋으로 알고리즘, 하이퍼파라미터의 최종 성능 평가

학습 데이터셋은 가중치 학습을 위한 데이터셋이고 검증 데이터셋은 하이퍼파라미터를 튜닝하여 아키텍처의 구성 및 동작방식을 결정합니다. 그리고 최종적으로 테스트 데이터셋으로 성능을 확인합니다.

이제 여러 홀드아웃 전략을 살펴봅시다.
- 단순 홀드아웃 검증
- K-겹 검증
- 반복 K-겹 검증

 > 홀드아웃 전략들에 대한 이미지를 찾아 보면서 쉽게 이해해보세요

 ### 단순 홀드 아웃 검증

 단순 홀드아웃 검증은 전체 데이터셋을 일정 비율로 데스트 데이터셋을 나눕니다. 데이터가 부족한 경우 30%를 학습에서 배제하면 학습 데이터가 부족해지는 상황이 발생하기 떄문에 데이터의 크기를 고려해 비율을 조정해야 합니다. 테스트 데이터셋을 나눈 후에는 알고리즘과 하이퍼파라미터가 고정될 때까지 테스트 데이터셋을 알고리즘과 격리해야 합니다. 또한 최적의 하이퍼파라미터를 찾기 위해 별도의 검증 데이터셋도 확보해야합니다.


 ### K-겹 검증

 데이터의 일부를 테스트셋으로 나눠 떼어내고, 나머지 데이터를 K개로 균등하게 분할합니다. k개로 분할한 각 블록을 겹(폴드)라고 합니다. 학습을 반복하면서 임의의 1개의 겹을 검증으로 사용하고 나머지 겹으로 학습을 진행합니다. 총 k번의 학습에서 얻은 모든 점수의 평균을 최종 점수로 합니다. k-겹 검증의 단점은 여러 조각으로 나뉜 데이터를 여러번 학습하기 때문에 연산량이 상당히 많아 진다는 점입니다.


 ### 반복 k-겹 검증
 머신 러닝 알고리즘을 한층 더 강건하게 만들기 위해 홀드아웃 검증 데이터셋을 만들 때 마다 데이터를 랜덤하게 섞을 수 있습니다.

## |데이터 전처리 및 특성 추출
신경망을 위한 데이터 전처리는 학습시킬 딥러닝 알고리즘에 적합한 데이터를 만드는 과정입니다. 다음은 데이터 전처리 과정에서 공통적으로 사용되는 항목입니다.

- 벡터화 (Vectorization)
- 정규화 (Normalization)
- 누락 데이터 처리 (Missing values)
- 특성 추출 (Feature extraction)

### 벡터화

학습하기 이전에 가장 먼저 해야 할 일은 데이터를 파이토치 Tensor로 변환하는 것입니다. 일반적인 이미지 데이터의 경우는 PIL Image나 numpy의 ndarray, pandas의 DataFrame 객체의 형태로 받아오는 경우가 많다. 이를 torchvision의 transforms.ToTensor()나  torch.from_numpy()와 같은 추상화된 라이브러리를 이용하면 됩니다. 사실 ㅌ테이블 형식의 구조화된 데이터라면 이미 벡터 형식이라고 할 수 있습니다. 이 경우 벡터 형식의 데이터를 앞서 언급한 도구를 이용하여 파이토치 텐서로 변환하기만 하면 됩니다.

### 정규화
우선 정규화를 하는 이유에 대해 간단한 예와 함께  설명해드리겠습니다. 나이 정보와 수입 정보를 통해 해당 사람이 사는 위치를 예측할 때 나이는 1 ~ 100사이의 값이고 수입은 10,000,000 ~ 100,000,000로 서로 다른 단위를 사용하고 완전히 다른 범위 값을 가지게 됩니다. 그렇게 되면 자연스레 값이 큰 수입 정보가 가중치에 큰 영향을 미치고 나이 정보는 무시되는 상황이 생깁니다. 따라서 알고리즘을 학습하기 전에 정규화를 적용하여 0 에서 1사이의 값을 갖도록 하는게 좋습니다. 또한 평균이 0이고 표준편차가 1인 데이터로 만드는 특성 정규화라는 방법도 있습니다.

### 특성공학
특성 공학은 머신 러닝 문제에 도메인 지식을 필요로 합니다. 즉 머신 러닝만 잘해서 되는게 아니라 해당 문제와 관련된 전문 지식도 알아야 된다는 것 입니다. 데이터가 많은 경우 딥러닝을 알고리즘을 사용해도 되지만 데이터가 부족한 경우에는 특성 공학에 집중하는 것이 바람직합니다. 

## |과대적합
앞에서 잠시 소개했듯이 과대적합은 인간에게 고정관념으로 비유할 수 있습니다. 모델이 훈련 데이터만 학습해서 다른 데이터를 만났을 때 제대로된 결과를 반환할 수 없는 경우입니다. 그렇다면 과대적합을 막기 위한 대표적인 기법을 알려드리겠습니다.

 - 많은 데이터 확보
 - 네트워크 크기 간소화
 - 가중치 규제 (Regularization)
 - 드롭아웃 적용

### 많은 데이터 확보

알고리즘이 학습을 위해 더 많은 데이터를 확보할 수 있다면, 데이터에 대한 일반적인 패턴을 학습함으로써 과대적합을 방지할 수 있습니다. 

컴퓨터 비전에서는 데이터 증식을 위해 Data Augmentation 기법을 사용합니다. 이미지를 회전, 자르기, 좌우 반전과 같은 방식으로 이미지를 약간 변형하여 데이터를 부풀리는 기술입니다. 또한 도메인에 대한 충분한 이해가 있다면 실제 데이터를 기반으로 합성 데이터를 만들어 쓸 수 있습니다.

### 네트워크 크기 줄이기

일반적으로 네트워크의 크기를 줄이면 용량도 줄어듦에 따라 학습 데이터셋의 과적합을 피할 수 있다. 그렇다고해서 무조건 네트워크 크기를 줄이는 것이 좋은 것은 아닙니다. 네트워크 크기가 크면 클수록 복잡한 것들을 표현할 수 있는데, 데이터가 부족하게 되면 오히려 과대적합을 야기할 수 있는 것입니다. 

### 가중치 규제(Regularization)

과대적합을 해결하는 핵심 원리는 단순한 모델을 구축하는 것입니다. 위에서 네트워크 크기를 줄임으로써 아키텍처의 복잡도를 줄일 수 있음을 알아보았습니다. 다른 방법은 네트워크의 가중치가 큰 값을 갖지 않도록 하는 것입니다.
규제 (Regularization)는 모델의 가중치가 큰 값을 가질 때 가중치 업데이트에 제약을 주어 원래보다 더 적은 값으로 업데이트가 되도록 합니다. 

가중치 규제 방법
1. L1 규제
2. L2 규제