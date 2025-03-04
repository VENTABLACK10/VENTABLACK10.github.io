---
layout: single
title:  "[Paper Reiview] Going deeper with convolutions"
---
## Going deeper with convolutions

### 0. Abstract
- Inception, Deep Comvolutional Neural Network 아키텍쳐 제안
- ImageNet 대규모 시각 인식 챌린지(ILSVRC14)에서 classification, detection 부문에서 좋은 성능 달성 및 기여
- 특징
  * 네트워크 내부에서 연산 자원을 효율적으로 활용
  * 연산 예산을 일정하게 유지하면서 네트워크의 깊이와 너비를 동시에 증가
  * 성능 최적화를 위해 아키텍처 설계는 Hebbian 원리와 Multi Scale Processing에 대한 직관을 기반 결정
  * 해당 아키텍쳐의 특정 구현은 모델을 GoogleNet이라 부르며, 22개의 layer로 이루어진 심층 신경망이다.

### 1. Introduction
- GoogleNet 모델은 2012년 ILSVRC 에서 우승했던 모델보다 12% 더 적은 parameters를 사용하면서, 더 높은 정확도를 기록했다.
- 본 논문에서 제안하는 심층 아키텍쳐는 단순히 정확도 향상에만 집중하는 것이 아니라, 실제 활용 가능성을 고려하여 설계되었다.
- 실험에서는 추론(Inference) 시, 150억 Multiply-Adds 내에서 연산을 수행하도록 제한해 대규모 데이터셋에서도 합리적인 비용으로 실용적으로 활용될 수 있도록 설계되었다.
- 본 논문에서 Deep이라는 단어는 두 가지 의미를 가진다.
  1. Inception 모듈이라는 새로운 조직 구조를 도입하여 네트워크 내부에 추가적인 계층 생성
  2. 네트워크 깊이(depth) 자체가 증가하여 보다 복잡한 표현 학습 가능
- ImageNet 대규모 시각 인식 챌린지(ILSVRC14)에서 classification, detection 부문에서 좋은 성능 달성 및 기여

### 2. Related Work
- LeNet-5와 같은 기존 CNN은 모델은 일반적으로 표준적인 구조를 가지고 있다.
  * 표준 구조: 연속적 Convolutional Layers -> 선택적 contrast Normalization & Max Pooling -> Fully-Connected Layers)
- Max-Pooling Layer의 Spatial Information 손실 우려에도 불구하고, 컴퓨터 비전 작업에서 성공적으로 활용
- 영장류의 Visual Cortex 모델에서 영감을 받아 다양한 크기의 고정된 Gabor Filters를 적용하여 다중 스케일을 처리 방법 제안 -> 이는 Inception 모델과 유사하지만, 해당 모델에서는 모든 필터가 학습되고 Inception 계층이 여러 번 반복되면서 22개의 층으로 구성된다. (GoogleNet)
- Network-in-Network: 1 X 1 합성곱 Layer를 추가하여 비선형성을 증가시키는 방법, 해당 계층 뒤에는 Relu 활성화 함수 적용
- 1 X 1 합성곱 역할: 차워 축소 모듈 기능, 연산 병목 제거(네트워크 크기 제한) -> 네트워크의 깊이, 너비 확장 가능

### 3. Motivation and High Level Considerations
- Deep Neural Networks 성능을 향상시키는 직접적인 방법은 네트워크의 크기를 증가시키는 것이다.
  1. 깊이(depth) 증가 -> 네트워크의 층(layers) 수 늘리기
  2. 너비(width) 증가 -> 각 층에서의 유닛(units) 수 늘리기
- 대량의 레이블된 학습 데이터 존재 시, 해당 방식은 더 높은 품질의 모델을 학습하는 쉬운 방법으로 여겨진다.
- 하지만, 위와 같은 해결법은 두 가지 주요 단점이 존재한다.
  1. 네트워크의 크기가 커질수록 Parameters 수도 증가하면서 Overfitting 가능성 증가 (병목현상)
  2. 연산 자원 사용량 증가
             
<br>           
            
- 위의 두가지 문제를 근본적으로 해결하는 방법이 Fully Connected 아키텍쳐에서 Sparsely Connected 아키텍쳐로 이동하는 것이다. (합성곱 계층 내부에서도 적용 가능)
- 이론적으로 Sparsely Connected 아키텍쳐가 과적합을 줄이고 연산량을 줄이는 데 유리하지만 실제로는 Non-uniform Sparse 데이터 구조에서 연산하는 것이 하드웨어적으로 비효율적이기 때문에 사용하지 않음

### 4. Architectural Details
- Inception 아키텍쳐는 합성곱 기반의 비전 네트워크에서 최적의 local sparse structure를 쉽게 이용할 수 있는 dense components로 근사화하고 커버할 수 있는지를 찾는 것에 기반한다.
- 현재 네트워크는 합성곱 계층으로 구성되며 translation invariance를 가정하고 있다.
- layer-by-layer 설계 방법은 이전 층의 correlation statistics를 분석한 후 높은 상관 관계를 가진 유닛들을 그룹화하는 방식으로 네트워크를 구성해야 한다고 주장한다. 이렇게 형성된 클러스터는 다음 층의 단위(unit)이 되고 이전 층의 유닛들과 연결된다. 여기서 각 유닛은 입력 이미지의 특정영역을 나타낸다고 가정하고, 이러한 유닛들이 filter bank로 묶인다.
- 입력에 가까운 하위 계층에서는 상관 관계가 높은 유닛들이 특정 local region에 집중되므로 다음 계층에서 1x1 합성곱을 적용하여 효과적으로 처리하도록 한다.
- 하지만, 일부 유닛들은 공간적으로 더 넓게 분포하려는 경향이 있어 더 큰 영역을 커버하는 합성고비 필요하다. 이에 크기가 더 큰 패치에 대한 합성곱이 적용되고 패치의 크기가 커질수록 적용되는 패치의 개수는 점점 줄어든다.
- 패치 정렬 문제를 피하기 위해, Inception 아키텍쳐는 1x1, 3x3, 5x5 크기만의 필터만을 사용하고 있다.
- 결과적으로 Inception 아키텍쳐는 이러한 다양한 크기의 필터를 활용한 여러 합성곱 계층을 병렬적으로 구성하고, 각 계층에서 나온 출력(filter banks)들을 하나의 출력 벡터로 병합하여 다음 단계의 입력으로 활용한다.              
![photo 225](/assets/img/blog/img225.png)                

- 또한, 각 Inception 모듈에 poolong 연산을 병렬적으로 추가하는 것이 성능 향상에 도움이 될 것으로 제안하고 있다. (Figure 2(a))
- Inception 모듈이 여러 층으로 쌓이면, 출력 간의 상관 통계는 변화한다. 상위 계층에서 더 추상적인 특징이 포착될수록, 특징들의 공간적 집주도는 감소할 것으로 예상되므로 상위 계층으로 갈수록 3x3, 5x5 합성곱의 비율을 증가시키는 것이 바람직하다.
- 그러나, 기본적인 형태의 Inception 모듈은 계산 비용이 높아질 수 있다는 문제가 있기 때문에 최적의 희소 구조를 커버할 수 있을지는 몰라도 비효율적으로 작동하여 계산량이 폭발적으로 증가하게 된다.
- 해당 문제를 해결하기 위해 계산 요구량이 과도하게 증가할 수 있는 경우 차원 축소 및 projection을 신중하게 적용하는 것이다. 가능한 많은 곳에서 희소 표현을 유지하면서 대량의 저보가 집계되기 위해 1x1 합성곱을 활용하여 계산량이 큰 3x3 및 5x5 합성곱을 적용하기 전에 차원을 줄인다.
- Inception Network는 위에서 설명한 Inception 모듈을 여러 층으로 쌓고, 일정한 간격으로 stride 2의 max pooling layer를 추가하여 grid의 해상도를 절반으로 줄이는 구조로 구성된다.
- 장점
  1. 각 단계에서 유닛의 개수를 대폭 증가시킬 수 있고 연산 복잡도가 통제 가능한 정도로 된다. -> 이는 차원 축소 이후 큰 패치 크기로 합성곱을 적용하는 방식 덕분이다.
  2. 시각적 정보는 다양한 스케일에서 처리된 후 통합되어야 하며, 이를 통해 다음 계층에서 여러 스케일의 특징을 동시에 추출할 수 있어야 한다는 직관적인 원칙과도 부합한다.
  3. 계산 차원의 효율적인 활용 덕분에, 각 단계의 너비를 증가시키거나 단계 수를 늘려도 심각한 계산량 문제가 발생하지 않는다.
  4. 비슷한 성능을 내는 기존 CNN 모델보다 2~3배 빠른 네트워크 설계했다.

### 5. GoogleNet
![photo 226](/assets/img/blog/img226.png)                 
- 해당 논문에서는 가자 성공적인 특정 사례인 GoogleNet의 세부사항을 Table 1에서 예시로 설명하고 앙상블에 포함된 7개 모델 중 6개는 동일한 네트워크 topology를 유짛면서, 서로 다른 샘플링 방식으로 훈련된 모델을 사용했다.
- Inception 모듈 내부를 포함한 모든 합성곱 계층은 Relu 활성화 함수를 사용한다.(차원 축소 및 projection layer 포함)
- 해당 네트워크의 receptive field shape은 224x224 이며, 입력 데이터는 RGB 색상 패널을 사용하고 mean subtraction을 수행한다.
- 3x3 reduce, 5x5 reduce는 각 합성곱 전에 사용되는 1x1 필터의 개수를 의미한다.
- pool proj는 max pooling 이후 projection 계층에서 사용되는 1x1 필터의 개수를 의미한다.
- 해당 네트워크는 연산 효율성과 실용성을 고려하여 설계되었기 때문에 메모리가 제한된 장치에서도 실행가능하도록 최적화되어 있다.
  - pooling layers 제외 네트워크 깊이: 22개 layers / 포함 시, 27개 layers
  - 전체 building blocks 100개
- average pooling과 함께 추가적인 linear layer를 포함하여 네트워크를 다른 레이블 집합에 맞게 쉽게 조정 및 미세 조정하도록 한다.
- 중간 계층에 보조 분류기를 축하여 초기 계층에서 더 판별력 있는 특징을 학습하도록 beh하고 역전파 과정에서 gradient signal을 증가시켜 학습 안정성을 향상시킨다. 또한, 추가적인 정규화 역할을 수행하여 과적합을 방지한다.
- 훈련 시, 보조 분류기의 손실은 네트워크 전체 손실에 가중치 0.3을 곱하여 추가되지만, 추론 단계에서는 해당 보조 네트워크가 제거된다.

<br>      

![photo 227](/assets/img/blog/img227.png)                   
- 보조 네트워크 구조
1. average pooling layer
  - filter size = 5x5
  - stride = 3
  - output size -> Inception (4a): 4x4x512 / Inception (4b): 4x4x5281
2. 1x1 convolution layer
  - filter = 128
  - 차원 축소 및 ReLU 활성화 함수 적용
3. Fully connecte layer
  - units = 1024
  - ReLU 활성화 함수 적용
4. dropout layers
  - dropout 비율 = 70%
5. linear layers, softmax classifier
  - softmax loss 적용
  - main classifier와 동일하게 1000개의 클래스 예측
  - 단, 추론 시 제거

### 6. Training Methodology
- 훈련과정에서 0.9 SGD 모멘텀 값을 사용
- 학습률을 8 epoch마다 4%씩 감소시키는 스케쥴러 적용
- 최종 추론 시 사용된 모델은 Polyak 평균화 기법을 활용하여 생성
- 이미지의 다양한 크기의 패치를 샘플링하는 방식이 효과적임을 발견 -> 이때 패치 크기는 전체 이미지 면적의 8%에서 100% 사이에서 균등하게 분포하며 종횡비는 3:4 ~ 4:3 사이에서 무작위로 선택
- 또한, Andrew Howard가 제안한 광학 왜곡이 어느 정도 과적합을 방지하는 데 유용하다는 것을 발견

### 7. ILSVRC 2014 Classification Challenge Setup and Results
- LSVRC 2014 Classification Challenge 목표: 이미지를 1,000개의 카테고리 중 하나로 분류
- 데이터셋 구성: training set 1,200,000개 / validation set 50,000개 / test set 100,000 개
- 성능 측정 지표
  - TOP-1 Accuracy: 모델이 예측한 가장 확률이 높은 클래스(1순위)가 정답과 일치하는 비율
  - Top-5 Error rate: 모델이 예측한 상위 5개 클래스 내에 정답이 포함되지 않는 경우의 비율 (순위를 결정하는 기준)
 
<br>            

- 접근 방식                  
![photo 228](/assets/img/blog/img228.png)                           
  1. GoogleNet 모델에 앙상블 적용
    - 동일한 GoogleNet 모델 7개를 독립적으로 훈련, 그 중 한가지는 더 넓은 버전으로 훈련
    - 모든 모델이 같은 초기화 값과 학습률을 사용했지만, 샘플링 방식 및 데이터 입력 순서를 다르게 사용
    - 테스트 시 앙상블 기법을 사용하여 성능을 향상
  2. 보다 공격적인 Cropping 기법 사용
    - 기존 cropping 방법(AlexNet 방식)보다 더 다양한 이미지 크기 활용
    - 4가지 크기의 이미지 resize => 높이 or 너비를 256, 288, 320, 352로 설정
    - 각 크기에서 3개의 위치 선택
      - 가로 이미지는 왼쪽, 중앙, 오른쪽
      - 세로 이미지는 위쪽, 중앙, 아래쪽)
    - 각 위치에서 다양한 crop 수행
      - 네 개 모서리 및 중앙 224 x 224 crop
      - 224 x 224로 resize한 이미지를 포함
      - 좌우 반전 적용
      - 총 144(4x3x6x2=144)개의 crop을 생성하여 테스트 수행
  3. Softmax 확률 정규화
    - 다수의 crop을 통해 얻은 softmax 확률 값을 평균하여 최종 예측 수행
    - 추가적인 대안 방법 실험
      - crop별 max pooling -> 성능 저하
      - 앙상블 모델 평균화만 수행 -> 성능 저하
    - 결과적으로, simple averaging이 가장 좋은 성능을 보임

<br>            

- 최종 성능 및 비교 분석
  - validation 및 test set에서 Top-5 errir rate 6.67% 달성 (대회 1위 기록)
  - 2012년 SuperVision(AlexNet, Top-5 오류율 16.4%) 대비 56.5% 상대적 감소
  - 2013년 Clarifai(Top-5 오류율 11.2%) 대비 약 40% 상대적 감소
  - SuperVision 및 Clarifai는 외부 데이터를 활용했지만, 해당 모델은 X
 - 결론
  - ILSVRC 2014에서 GoogLeNet을 활용한 모델 앙상블과 고급 크롭핑 기법을 통해 최고 성능(Top-5 오류율 6.67%)을 달성
  - 기존 방식보다 더 많은 crop을 사용하고, softmax averaging를 활용하는 것이 효과적임을 증명
  - 외부 데이터를 사용하지 않고도 기존 모델보다 성능을 크게 향상

<br>           

- 추가 실험 및 모델 평가
  - 모델 개수 및 crop 개수 변화에 따른 성능 분석 수행
  - 하나의 모델을 사용할 때는 validation set에서 Top-1 error rate가 가장 낮은 모델 선택
  - 과적합을 방지하기 위해 validation set 성능을 측정하여 모델 성능 평가

### 8. ILSVRC 2014 Detection Challenge Setup and Results
- ILSVRC 2014 Detection Challenge 목표: 이미지에서 200개 클래스 중 하나에 속하는 객체의 Bounding Box 검출
- 검출 기준
  - 예측된 객체가 정답 클래스와 일치해야 한다.
  - 예측된 바운딩 박스가 IoU ≥ 50% 기준을 충족해야 한다.
  - 불필요하거나 잘못된 객체 검출은 False Positive(위양성)로 간주하여 패널티를 받는다.
- Classification Challenge와 차이점
  - 이미지 내에 객체가 여러 개 포함 가능하다.
  - 일부 이미지에는 객체가 없을 수도 있다.
  - 객체 크기가 매우 작거나 클 수 있다.
- 평가 지표: mAP(평균 정밀도)를 사용하여 성능을 평가한다.

<br>        

- 접근 방식: 기존의 R-CNN 방식을 기반으로 하여 개선
  1. GoogLeNet을 객체 검출에 활용
    - R-CNN 방식과 유사하게, 지역(region) 기반 검출을 수행한다.
    - Inception 모듈을 사용한 GoogLeNet을 영역 분류기(region classifier)로 적용한다.
  2. 영역 제안(Region Proposal) 방식을 개선
    - 기존 R-CNN의 Selective Search 방식을 유지하면서, MultiBox 기법을 추가 적용 -> 객체를 올바르게 감지할 확률(recall) 증가
    - Selective Search의 superpixel 크기를 2배로 증가시켜 False Positive를 줄여 제안된 Bounding box 개수를 줄였다.
    - MultiBox에서 추가적인 200개 영역을 제안하여 전체적으로 R-CNN보다 60% 적은 영역 제안을 사용하면서 coverage(객체 검출율)는 92% → 93%로 증가했다.
    - 결과적으로, 단일 모델 기준 mAP가 1% 향상
  3. ConvNet 앙상블 적용
    - 하나의 모델만 사용하는 대신, 6개의 ConvNet을 앙상블하여 각 객체의 영역을 분류한다.
    - 앙상블 적용 전 성능: 40% mAP -> 앙상블 적용 후 성능: 43.9% mAP => 향상
  4. Bounding Box Regression 적용 X
    - R-CNN과 다르게 Bounding Box의 위치를 더 정확히 보정하는 Bounding Box Regression 기법을 적용하지 않았다.(시간 부족)

<br>           

![photo 229](/assets/img/blog/img229.png)                     
- 최종 성능 분석
  - 2013년 대비 정확도 2배 증가
  - 상위권 팀들은 모두 CNN 기반 방법 사용
  - 대회 성적은 Table 4 참고
  - GoogLeNet 팀은 Localization 데이터로 사전 훈련하지 않음
- 단일 모델 성능 비교(Table 5 참고)
  - 단일 모델 기준 최고 성능은 Deep Insight 팀이 기록 -> 그러나, Deep Insight 팀은 3개 모델을 앙상블했을 때도 성능이 0.3점만 향상
  - 반면, GoogLeNet은 앙상블을 적용할수록 성능이 크게 향상됨
- 결론
  - GoogLeNet은 R-CNN을 기반으로 객체 검출에 최적화된 Inception 모델과 개선된 Region Proposal 방식을 결합하여 2014년 ILSVRC Detection 부문에서 강력한 성능을 기록
  - 특히, Selective Search + MultiBox 결합, Superpixel 크기 증가, ConvNet 앙상블 활용을 통해 mAP를 높이는 데 성공
  - 단일 모델과 비교했을 때 앙상블 효과가 매우 크다.

### 9. Conclusions
- 최적의 희소 구조를 기존의 밀집된 building blocks으로 근사하는 것이 컴퓨터 비전 신경망 개선하는 데 있어 유효한 방법임을 입증
- 해당 방법의 주요 장점은 더 얕고 좁은 네트워크에 비해 연산 요구량의 증가를 최소화하면서 성능을 크게 향상시킬 수 있다는 점이다.
- 객체 탐지 모델의 경우, context 정보와 bounding box regression을 수행하지 않았음에도 불구하고 경쟁력 있는 성능을 보였다.

