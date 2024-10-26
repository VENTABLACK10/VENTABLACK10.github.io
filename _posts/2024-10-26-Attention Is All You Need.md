---
layout: single
title:  "[Paper Reiview] Attention Is All You Need"
---
## Attention Is All You Need

### 1. Abstract
* 기존의 sequence model은 encoder, decoder를 포함하는 복잡한 RNN, CNN model를 기반으로 한다.
* 본 논문은 Attention 매커니즘을 이용하여 recurrent 구조와 convolution 구조를 완전히 제거한 Transformer model을 제안한다.
* 해당 Transformer model은 기존 모델에 비해 훈련 비용이 적고 성능이 뛰어나다.

### 2. Background
* 기존 Extended Neural GPU, ByteNet, ConvS2는 CNN을 사용해 입력과 출력 위치에 대해 병렬 계산을 수행하지만, 위치 간 거리가 멀수록 신호를 연결하기 위한 연산 수가 증가한다.
* 본 논문의 Transformer model은 병렬 계산 수를 상수로 줄이고 멀리 떨어진 위 간의 의존성 학습을 개선했다.
* 또한, Transformer model은 RNN or CNN을 사용하지 않고 self-attention만으로 입력 및 출력의 표현을 계산하는 최초의 모델이다.

### 3. ModelArchitecture
* 본 논문의 Transformer model은 encoder, decoder 모두에서 stacked self-attention과 point-wise FC Layer를 사용한다.         
![photo 155](/assets/img/blog/img155.png)                            
<br>        

#### 3.1 Encoder and Decoder Stacks
* Encoder: N=6인 6개의 동일한 layer로 구성되고 각 층은 multi-head self-attention과 위치별 feed-forward network라는 두 개의 sub layer를 포함한다. 또한, Residual Connection과 Layer Normalization을 통해 d_model의 차원이 512인 출력을 생성한다.
* Decoer: encoder와 유사하게 N=6인 6개의 층으로 구성되고 인코더 출력에 대한 multi-head attention sub layer를 추가한다. 또한, self-attention 이후 위치에 대한 접근을 막기 위해 masking을 사용하고 이는 i번째 예측 이전 정보만을 이용해 예측하도록 도와준다.

#### 3.2 Attention
* Attention Function은 Query, Key-Value 쌍을 받아서 Query와 각 Key 간의 유사도를 통해 계산된 가중치를 사용해 Value들의 가중 합을 출력한다.                  
![photo 156](/assets/img/blog/img156.png)               
<br>                                  
       
#### 3.2.1 Scaled Dot-Product Attention             
  ![photo 157](/assets/img/blog/img157.png)               
* Input: d_k 차원의 Query, Key / d_v 차원의 Value
* Attention = Query와 Key의 전치 행렬 값과의 곱 -> Scaling을 위해 √dk로 나누기 -> Softmax 함수 적용 -> V와 행렬 곱
* Scaled Dot-Product Attention은 /√dk로 나누는 scaling을 통해 최적화된 행렬 곱세믈 사용할 수 있고 이는 공간 효율적이며 빠르다.
* 또한, d_k값이 큰 경우, softmax 함수의 기울기 소실 문제를 해결해준다.

#### 3.2.2 Multi-Head Attention          
  ![photo 158](/assets/img/blog/img158.png)                
* Multi-Head Attention은 Query, Key, Value을 각각 d_k, d_k, d_v 차원으로 선형 변환한 후, 이를 h번 병렬로 attention 함수를 수행항여 최종 값을 생성한다.
* 이를 통해 서로 다른 위치에서 다양한 정보를 통시에 처리할 수 있게 되었다.
* 본 논문에서는 h=8개의 head를 사용하고 각 head의 차원을 줄임으로써 single attention과 비슷한 계산 비용을 유지한다.
  
#### 3.2.3 Applications of Attention in our Model
* encoder-decoder attention: Query는 decoder에서 key, Value는 encoer에서 가져와 decoder가 input sequence의 모든 위치에 주목할 수 있게 한다.
* encoder의 self attention: self-attention layer에서는 key, Value는, Query가 모두 동일한 위치에서 오고 encoder의 각 위치가 이전 layer의 모든 위치를 참조한다.
* decoder의 self attention: decoder의 각 위치가 해당 위치까지의 모든 decoder 위치에 주목할 수 있게 한다. 또한, decoder에서 auto-regressive 특성을 유지하기 위해 모든 값을 -∞로 설정하여 masking하는 방식을 활용한다.           
<br>                  

#### 3.3 Position-wise Feed-Forward Networks              
  ![photo 159](/assets/img/blog/img159.png)                                   
* encoder와 decoder의 각 층에는 위치별로 동일하게 적용되는 FC Feed Forward Network(FFN)가 포함되어 있다.
* FFN은 2개의 선형 변환과 그 사이에 ReLU 활성화 함수를 포함한다.
* 선형 변환은 위치마다 동일하지만, 각 층마다 서로 다른 파라미터를 사용한다.
* 본 논문에서는 입력과 출력의 차원은 d_model=512 사용했고 내부 layer의 차원은 d_ff = 2048로 사용했다.             
<br>                 

#### 3.4 Embeddings and Softmax
* Transformer model은 입력과 출력 token을 d_model 차원의 벡터로 변환하기 위해 학습된 embedding을 사용한다.
* 또한, decoder의 출력은 선형변환과 softmax 함수를 통해 다음 token의 확률로 변환한다.
* 해당 과정에서 두 embedding layer와 softmax 이전의 선형 변환에 동일한 가중치 행렬을 공유하고, embedding layer에서는 가중치에 √d_model을 곱해 사용한다.               
![photo 160](/assets/img/blog/img160.png)                      
* Self-Attention은 멀리 떨어진 정보의 상호작용을 빠르게 학습하기 때문에 장기 의존성 학습에 유리하다.
* 또한, 병렬 처리에도 강점을 보이기 때문에 긴 sequence를 다루는 작업에서 다른 모델에 비해 우수한 성능을 보인다.

#### 3.5 Positional Encoding
* 본 논문의 Transformer model에는 recurrent or convolution이 없기 때문에, sequence의 순서를 모델이 활용할 수 있도록 sequence 내 token의 상대적 or 절대적 위치에 대한 정보를 추가한다. 이를 positional encodings라고 한다.
* positional encoding은 embedding과 동일한 d_model의 차원을 가지기 때문에 합칠 수 있다.                   
![photo 161](/assets/img/blog/img161.png)                    
* positional encoding은 위의 식과 같이 삼각함수와 같은 주기함수를 통해 구현할 수 있다.
* 삼각함수를 사용하는 이유는 모델이 상대적 위치 정보를 쉽게 학습할 수 있도록 하기 위해서다.
* pos는 위치를 의미하고 i는 차원을 의미한다. 

### 4. Why Self-Attention
![photo 160](/assets/img/blog/img160.png)          
* Self-Attention은 일정한 수의 순차적 연산으로 모든 위치를 연결할 수 있어 병렬화가 용이하고 sequence의 길이가 expression 차원보다 짧을 때 계산 효율이 높다.
* 반면, recurrent layer는 O(n)의 순차적 연산이 필요해 긴 sequence에서 비효율적이고 convolutuional layer는 모든 위치 간의 직접 연결을 위해 여러 layer가 필요하고 계산 비용이 크다.
* Self-Attention은 더 해석 가능한 모델을 제공할 수 있고, 개별 attention head가 문장의 구문 및 의미 구조를 학습할 수 있다.

### 5. Training
1. Training Data and Batching
  * WMT 2014 dataset을 사용해 모델 훈련
  * Byte-Pair Encoding(BFE)의 37,000개의 공유된 어휘 구성
  * English-French의 32,000개 word-piece 어휘 사용
  * 각 training batch에 25000개의 소스token과 25000개의 타겟token 포함
2. Hardware and Schedule
  *  8개의 NVIDIA P100 GPU 사용 단일 기기에서 훈련
  *  Base Model 12시간훈련 / Big Mode l35일 훈련
3. Optimizer
  *  Adam Optimizer 사용
  *  hyperparameter: β1=0.9, β2=0.98=0.98, ϵ=10^−9
4. Regularization
  * Residual Dropout: 각 sub layer 출력에 dropout 적용, encoder 및 decoder embedding 합계에도 적용
  * Label Smoothing: 0.1로 설정, 불확실한 예측 학습을 통해 정확도와 BLEU Score 향상                      
![photo 162](/assets/img/blog/img162.png)                         
* 해당 table을 통해 Transformer model이 다른 model에 비해 BLEU Score가 높고 계산 비용이 적은 것을 알 수 있다.

### 6. Results
* 다음은 Transformer의 다양한 구성 요소의 중요성을 평가하기 위해 모델을 여러 방식으로 변형하여 성능 변화를 측정했다.          
![photo 163](/assets/img/blog/img163.png)            
* (A) = attention head 수 및 차원 변화 -> 단일 head에서만 성능 저하 발생 및 8개의 head에서 최적 성능
* (B) = attention key 크기 감소 -> d_k 값이 클수 모델 성능에 긍정적 영향을 미침
* (C) = 모델 크기 증가 -> BLEU 점수는 향상됐지만, 훈련 비용도 함께 증
* (D) = drop-out 변화 ->  drop-out 적용이 성능 개선에 도움
* (E) = 위치 인코딩 변경 -> 기본 설정과 유사한 성능 
* big = d_model=1024, d_ff=4096, h=16 설정 -> 성능 대폭 향상
  ![photo 164](/assets/img/blog/img164.png)                     
* 영문 구문 분석 작업 및 semi-supervised 학습에서도 기존 RNN 기반 모델들보다 뛰어난 성능 발휘했디.

### 7. Conclusion
* 해당 논문은 Transformer Model이 attention 매커니즘만으로 구성된 최초의 sequence 변환 Model임을 보인다.
* 본 논문의 Transformer Model은 기존의 encoder-decoder 구조에서 사용되던 RNN Layer를 multi-head self attention으로 대체했다.
* 또한, 기계 번역 작업에서 RNN, CNN 기반 모델보다 빠른학습 및 높은 성능을 나타냈다.
