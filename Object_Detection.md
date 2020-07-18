# Object Detection

본 글은 객체 탐지(Object detection)에 대해 공부하면서 정리한 내용입니다.

Object Detection 특성상 이미지, 영상을 다루기 때문에 CNN에대한 기초지식이 있어야한다는 점을 당부 드립니다.



object detecton 관련 정보

* R-CNN
* Fast R-CNN
* Faster R-CNN
* SSD
* RPN

객체 탐치는 이미지, 영상 속의 어떤 객체(Label)가 어디에(x,y) 어느 크기로(w,h) 존재하는지를 찾는 Task를 얘기합니다. 



## R-CNN

1. Hypothesize Bounding Boxes (Proposals)
   * Image로부터 Object가 존재할 적절한 위치에 Bounding Box Proposal을 실행한다.
   * 2000개의 Proposal이 생성된다.
2.  Resampling pixels / features for each boxes
   * 모든 Proposal을 Crop하고 동일한 크기로 만든다.
3. Classifier / Bounding Box Regressor
   * 이미지를 Classifier와 Bounding Box Regressor로 처리한다.



R-CNN의 문제점은 바로 한 이미지에서 2000개의 Proposal을 전부 CNN연산을 해야한다는 점입니다.

그래서 제안한 방식은 이미지 하나를 CNN연산을 거쳐 feature map을 만든 다음, 이 feature map에서 crop을 진행하는 방식입니다. 이를 Fast R-CNN이라고 부릅니다.



## Fast R-CNN

Fast R-CNN은 R-CNN이 모든 Proposal에 대해서 네트워크를 거쳐야 하는 병목(bottleneck) 구조의 단점을 개선하고자 제안된 방식입니다.

가장 큰 차이점은 각 Proposal이 아니라 전체 이미지에 대해 CNN을 한번 거친 후 출력 된 feature map에서 객체 탐지를 수행한다는 점입니다.

하지만 Fast R-CNN에서는 Region Proposal을 할 때 CNN Network가 아닌 Selective search 외부 알고리즘을 수행하여 병목현상이 발생하게 됩니다.



## Faster R-CNN

CNN 외부에서 사용되던 알고리즘인 Region Proposal을 RPN(Region Proposal Network)네트워크를 이용하여

CNN 내부에서 모든 수행을 마치도록(end-to-end) 한다. 그래서, 병목현상을 해소했다고 볼수 있다.

* end-to-end란 학습을 위해 여러 단계가 필요한 처리과정을 한번에 처리하는 것을 말합니다. 즉, 데이터만 입력하면 원하는 목적을 학습하게 되는 것입니다.

1. 기존의 Selective  search가 아닌 CNN(RPN)로 해결
2. CNN을 통과한 Feature map에서 슬라이팅 윈도우를 이용해 각 지점(anchor)마다 바운딩 박스와 좌표와 점수를 계산
3. 2:1, 1:1. 1:2의 종횡비(aspect ratio)로 객체 탐색

쉽게 말해서 object detection을 위한 통합된 네크워크인 것이다.





