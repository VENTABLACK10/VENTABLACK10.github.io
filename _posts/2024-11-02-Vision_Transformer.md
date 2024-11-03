---
layout: single
title:  "[Paper Reiview] An Image Is Worth 16 X 16 Words: Transformers For Image Recognition At Scale"
---
## An Image Is Worth 16 X 16 Words: Transformers For Image Recognition At Scale

### 1. INTRODUCTION
* Transformer architecture는 NLP의 주요 모델이 되었고 대규모 텍스트 데이터로 사전 학습 후, 작은 데이터셋으로 미세 조정하는 방식이 일반적으로 자리 잡았다.
* 당시, Computer Vision에서는 여전히 CNN이 지배적이었고 CNN과 self-attention을 결합하려는 연구가 지속적으로 진행되었지만 여전히 기존 CNN 모델의 성능이 더 우세했다.
* NLP의 Transformer에서 영감을 받아 이미지 분류 작업에 최소한의 수정만 가한 표준 Transfomer을 적용했다.
* 이미지를 패치로 나누고 해당 패치들을 Transformer에 입력하여 지도 학습을 수행했다.
* Vision Transfomer Model은 대규모 데이터셋으로 학습할 수록 일반화 성능이 향상되었다.

### 2. RELATED WORK
* Transformer는 NLP에서 최첨단 방법이 되었고 BERT와 GPT 모델은 각각 self-supervised와 language modeling을 통해 사전 학습 후 특정 작업에 맞게 미세 조정된다.
* 이미지에 self-attention을 단순하게 적용하면 계산 비용이 높기 때문에 이를 해결하기 위한 여러 방법이 제안되었다.
* 일부 연구는 각 픽셀의 국소 영역에서만 self-attention을 적용하거나, Sparse Transformer를 사용해 이미지에 global self-attention을 효율적으로 적용하려 했다.
* Cordonnier et al. (2020)의 모델은 ViT와 유사하게 2x2 패치를 사용하지만, 이 모델은 작은 해상도에만 적합하다.
* 본 논문의 연구는 더 큰 데이터셋을 사용한 대규모 사전 학습이 ViT 성능을 CNN과 유사한 성능으로 만든다
* CNN과 self-attention을 결합하려는 다양한 연구가 있었고 이러한 접근 방식은 이미지 분류, 물체 감지, 비디오 처리 등에서 유망한 결과를 보여줬다.
* iGPT는 이미지 해상도와 색 공간을 축소하여 Transformer를 적용한 모델로, 비지도 학습을 통해 이미지 분류에서 72% 정확도를 달성했다.
* 본 논문의 연구는 더 큰 데이터셋을 사용한 이미지 인식 성능을 탐구하고 ImageNet-21k 및 JFT-300M에서 Transformer를 훈련하여 CNN 기반 모델과 비교해 성능을 확인한다.

### 3. METHOD
* 모델 설계 시, 본래의 Transfomer 구조를 따라서 NLP Transformer 아키텍처와 효율적인 구현을 거의 그대로 사용할 수 있었다.
* 모델의 개요는 아래 그림과 같다.                
![photo 165](/assets/img/blog/img165.png)          

#### 3.1 VISION TRANSFORMER (VIT)
1. Input Processing
  * 표준 Transformer는 1D token embedding sequence를 입력으로 받지만, ViT는 2D 이미지를 입력으로 받기 위해 이미지를 2D 패치로 나눈다.
  * 원본 이미지 x∈R^H×W×C 를 패치 sequence x_p∈R^N×(P^2⋅C)로 변환한다.
  * H x W : 이미지 해상도 | C: 채널 수 | P x P: 패치의 해상도 | N = (H x W)/P^2 : 패치 수(Transformer의 입력 시퀸스의 길이)
  * 각 패치는 고정된 크기의 latent vector D로 매칭되고 이를 통해 patch embedding을 생성한다.
2. Prepending a learnable class embedding
  * 각 패치 임베딩 시퀸스에 학습 가능한 class embedding을 앞에 추가한다.
  * Transformer의 출력을 통해 얻은 class embedding이 이미지 전체의 표현 y로 사용된다.
  * 해당 embedding에 classification head가 연결되고 pre-training 시에는 MLP로 fine-tuning 시에는 single linear layer로 구현된다.
3. Positional Embedding
  * 위치 정보를 유지하기 위해 positional embedding을 patch embedding에 추가한다.
  * 1D positional embedding을 사용하고 2D positional embedding을 사용해도 성능에는 큰 차이가 없음을 논문에서 언급했다.
4. Transformer Encoder
  * Transformer 인코더는 다층 self-attention과 MLP 블록으로 구성된다.
  * 각 블록 앞에는 정규화(layernorm)가 적용되고 잔차 연결(residual connection)이 뒤에 추가된다.
  * MLP는 GELU 활성화 함수를 사용하는 2개의 layer로 구성된다.                
  ![photo 166](/assets/img/blog/img166.png)        
5. Inductive Bias
  * ViT는 CNN에 비해 이미지에 대한 유도 편향이 적다.
  * CNN은 각 레이어에서 지역성, 2D 이웃 구조, translation equivariance을 활용하지만, ViT의 self-attention layer에는 전역적으로 작동하고 2D 공간 관계를 초기 위치 embedding에서 따로 제공하지 않는다.
  * 이러한 공간 관계는 모델이 학습을 통해 습득한다.
6. Hybrid Architecture
  * ViT는 CNN의 feature map에서 패치를 추출하여 input sequence를 구성할 수 있다.
  * 해당 경우, CNN feature map에서 추출된 패치에 patch embedding projection E를 적용한다.
  * 1x1 크기의 패치를 사용할 경우, feature map의 공간 차원을 평탄화하여 Transformer input sequence를 얻는다.

#### 3.2 FINE-TUNING AND HIGHER RESOLUTION
* ViT는 큰 데이터셋으로 사전 학습을 하고 이후 작은 데이터셋을 사용하는 downstream 작업에 fine-tunning을 한다.
* 이를 위해 사전 학습된 예측 head를 제거하고, downstream class의 수를 K로 하는 0으로 초기화된 D x K FeedForward layer를 추가한다.
* fine-tunning 시 pre-training보다 더 높은 해상도에서 작업하는 것이 유리할 때가 있다.
* 높은 해상도의 이미지를 입력으로 사용할 때, 패치 크기는 동일하게 유지하여 더 큰 sequence 길이를 생성한다.
* ViT는 메모리 제약 내에서 임의의 sequence 길이를 처리할 수 있지만, 해당 경우는 사전 학습된 positional embedding의 의미가 상실될 수 있다.
* 전 학습된 positional embedding의 의미가 상실되는 것을 방지하기 위해 원본 이미지에서의 위치를 기준으로 2D interpolation을 사용해 사전 학습된 positional embedding을 보정한다.
* fine-tunning 시 더 높은 해상도를 위한 조정 과정이 ViT에서 2D 이미지 구조에 대한 Inductive Bias을 수동으로 적용하는 유일한 지점이다.

#### 4. EXPERIMENTS
* ResNet, Vision Transformer(ViT), Hybrid model의 표현 학습 능력을 평가한다.
* 각 모델이 필요로 하는 데이터 양을 이해하기 위해 다양한 크기의 데이터셋에서 사전 학습을 진행하고 여러 benchmarks 과제를 평가한다.
* 모델의 사전 학습에 소요되는 계산 비용을 고려할 때, ViT는 매우 우수한 성능을 보여준다. (계산 효율성이 좋다.)
* self-supervision을 이용한 작은 실험을 수행한 결과, self-supervision을 적용한 ViT가 미래에 대한 가능성을 보여준다

#### 4.1 SETUP          
![photo 167](/assets/img/blog/img167.png)              
1. Datasets               
  * ImageNet: 1,000개의 클래스, 130만 개의 이미지
  * ImageNet-21k: 21,000개의 클래스, 1,400만 개의 이미지
  * JFT: 18,000개의 클래스, 3억 300만 개의 고해상도 이미지
  * Data preprocessing
    * downstream 작업의 테스트 세트와 중복되지 않도록 데이터셋을 정리한다.
  * 평가에 사용된 benchmarks
    * mageNet, ReaL label, CIFAR-10/100, Oxford-IIIT Pets, Oxford Flowers-102.
  *VTAB classification suite
    * 19개의 다양한 작업에서 모델의 성능 평가
  *  전이 학습 성능 평가
    *  각 작업당 1,000개의 학습 예제를 사용하여 적은 데이터로 전이 학습 성능을 평가
    * task groups: Natural(일반 이미지) Specialized(의료 및 위성 이미지) Structured(기하하적 이해가 필요한 이미지)
2. Model Variants
  * ViT는 BERT의 설정을 기반으로 “Base,” “Large,” 및 “Huge” 모델을 사용한다.
  * example: ViT-L/16은 Large 모델로 16x16 패치 크기를 의미
  * Transformer의 시퀀스 길이는 패치 크기에 반비례하므로, 패치 크기가 작을수록 계산 비용이 증가한다.
  * 기준 CNN: 수정된 ResNet(BiT) -> Group Normalization과 표준화된 컨볼루션을 사용하여 전이 성능을 향상시킨다.
  * Hybrid model: 중간 feature map을 ViT에 1 픽셀 패치로 입력하고, 시퀀스 길이를 늘리기 위해 ResNet의 일부 레이어 구성을 변경한다.
3. Training & Fine-tuning
  * 모든 모델은 Adam 옵티마이저로 학습, hyperparameter = {batch_size= 4096. weight decay=0.1, β_1=0.9, β_2=0.999}
  * fine tunning 시 batch_size=512로 SGD와 momentum을 사용하고 일부 모델의 경우 높은 해상도로 fine tunning을 진행한다.
4. Metrics
  * 각 모델의 fine tunning 후, accuracy와 few-shot accuracy를 통해 downstream 작업 성능을 평가하고 few-shot accuracy는 빠른 평가가 필요할 때 사용한다.

#### 4.2 COMPARISON TO STATE OF THE ART
1. ViT-H/14, ViT-L/16 vs CNN model(Big Transfer (BiT), Noisy Student:)
  * Big Transfer (BiT): 대형 ResNet 기반 지도 학습 모델
  * Noisy Student: EfficientNet 기반 반지도 학습 모델로, ImageNet에서 최첨단 성능을 기록
  * 모든 모델이 TPUv3 하드웨어에서 학습되었고 TPU 코어 수와 학습 시간을 곱한 TPUv3-core-days로 계산 비용을 평가한다.    
  ![photo 168](/assets/img/blog/img168.png)               
  * JFT-300M로 사전 학습한 ViT-L/16: BiT-L을 모든 작업에서 능가, 계산 자원도 적게 소모
  * ViT-H/14: 높은 성능, 특히 ImageNet, CIFAR-100, VTAB와 같은 어려운 데이터셋에서 우수
  * ImageNet-21k로 사전 학습한 ViT-L/16: 자원 소모가 적고 우수한 성능, 8 TPUv3 코어로 약 30일 동안 학습 가능     
  ![photo 169](/assets/img/blog/img169.png)        
  * ViT-H/14가 Natural 및 Structured 작업에서 다른 모델들보다 뛰어난 성능을 나타낸다.
  * Specialized 작업에서는 BiT-R152x4와 비슷한 성능을 보인다.

#### 4.3 PRE-TRAINING DATA REQUIREMENTS
* Vision Transformer(ViT)의 성능이 데이터셋 크기에 따라 어떻게 변하는지 평가하기 위해 두 가지 실험 수행            
  ![photo 170](/assets/img/blog/img170.png)   
* 첫 번째 실험: 데이터셋 크기 증가(ImageNet → ImageNet-21k → JFT-300M)로 사전 학습 후 ImageNet으로 fine-tunning
  * 작은 데이터셋에서는 ViT-Large가 ViT-Base보다 성능이 떨어졌으나, 큰 데이터셋(JFT-300M)에서는 큰 모델이 완전한 성능을 발휘
  * BiT CNN은 작은 데이터셋에서 ViT보다 우수했으나, 큰 데이터셋에서는 ViT가 더 나은 성능 발휘
* 두 번째 실험 : JFT-300M의 하위 집합(9M, 30M, 90M)과 전체 데이터셋을 사용하여 모델 학습.
  * 정규화 없이 동일한 하이퍼파라미터로 실험하여 모델의 고유 특성을 평가했다.
  * ViT는 작은 데이터셋에서 과적합이 더 심하지만, 큰 데이터셋에서는 ResNet을 능가한다.
  * 동일한 계산 비용 내에서 Vision Transformer(ViT)가 ResNet보다 일반적으로 더 우수한 성능을 보임.
* 작은 데이터셋에서는 CNN의 유도 편향이 유리하지만, 큰 데이터셋에서는 ViT가 직접 패턴을 학습하여 더 좋은 성능을 보인다.
* ViT의 few-shot 전이 성능이 유망하며, 향후 연구로 추가 분석이 기대됨을 언급했다.
#### 4.4 SCALING STUDY
  * JFT-300M에서 다양한 모델의 전이 성능을 평가하여 계산 비용 대비 성능을 비교한다.
  * Model set: 7 ResNet, 6 ViT, 5 Hybrid
  ![photo 171](/assets/img/blog/img171.png)
  * ViT는 ResNet보다 계산 효율성이 높다.
  * 하이브리드 모델은 작은 계산 비용에서는 ViT보다 성능이 우수하지만, 모델이 커질수록 차이가 사라진다.
  * ViT는 실험 범위 내에서 성능 포화가 나타나지 않아 더 큰 확장이 가능함을 시사한다.
#### 4.5 INSPECTING VISION TRANSFORMER
* Vision Transformer(ViT)의 첫 번째 layer는 평탄화된 패치를 저차원 공간으로 선형 투사하고 학습된 embedding filter의 주성분은 각 패치의 세부 구조를 나타내는 기저 함수와 유사한 형태를 보인다.
* projection 이후, 학습된 positional embedding이 패치 표현에 추가되고 모델은 가까운 패치들이 더 유사한 positional embedding을 가지도록 학습한다.
* 행-열 구조가 나타나고 더 큰 grid에서는 sin 함수 형태의 구조도 보이며 이로 인해 2D Embedding 변형이 성능 향상에 기여하지 않는다.
* Self-attention을 통해 ViT는 이미지 전반에 걸쳐 정보를 통합할 수 있다.
* 일부 attetnion head는 최하위 layer에서도 전역적인 정보를 통합하고, 다른 head는 국소화횐 정보를 유지한다.
* hybrid model에서는 국소화된 attention이 덜 두드러지고 네트워크 깊이에 따라 attention distance가 증가한다.
* 전체적으로 model은 분류에 의미적으로 중요한 이미지 영역에 주의를 기울이고 있음을 확인했다.
![photo 172](/assets/img/blog/img172.png)

#### 4.6 SELF-SUPERVISION
* BERT에서 사용된 masked language modeling task를 모방하여 masked patch prediction을 통한 자가 지도 학습을 예비적으로 탐구했다.
* 자가 지도 학습을 통해, ViT-B/16 소형 모델이 ImageNet에서 79.9%의 정확도를 달성했다.
* 이는 무작위 초기화에서 학습한 경우에 비해 2% 유의미한 개선이지만, 지도 학습 사전 학습에 비해서는 아직 4% 뒤쳐진 성능이다.
![photo 172](/assets/img/blog/img173.png)
* 왼쪽: ViT-L/32 모델의 초기 RGB 값에 대한 linear embedding filter
* 중앙: ViT-L/32 모델의 positional embedding 유사도, 코사인 유사도로 표시된 각 패치 간 위치 유사성
* 오른쪽: 각 layer에서 attended area 크기와 네트워크 깊이 간 관계, 각 점은 특정 layer와 head의 평균 attention distance를 의미한다.
  
### 5 CONCLUSION
* Vision Transformer(ViT)를 이미지 인식에 적용하면서 이미지 특화 inductive biases 없이 이미지를 패치 sequence로 해석하여 NLP에서 사용하는 표준 Transformer encoder로 처리했다.
* Vision Transformer(ViT)는 대규모 데이터셋에서 사전 학습할 때 매우 효과적이고, 여러 이미지 분류 작업에서도 좋은 성능을 발휘한다.
