# 오버피팅 대처법 정리
1. 데이터 수를 늘린다.
2. 데이터 어그멘테이션 기법을 사용한다.
3. L1, L2 정규화를 수행한다.
    - L1은 Taxicab geometry로 미분이 불가능하지만 이상치에 강하다
    - L2는 유클리드 거리로 미분이 가능지만 L1에 비해 이상치에 약하다.
4. 배치 정규화를 수행한다.
    - internal covarient shift 문제를 해결
    - whitening(백색소음)이 바이어스(b)를 무시하게 된다는 점을 해결
5. 드롭아웃 기법을 사용한다.