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
#### 3.1 VISION TRANSFORMER (VIT)

#### 3.2 FINE-TUNING AND HIGHER RESOLUTION

#### 4. EXPERIMENTS

#### 4.1 SETUP

#### 4.2 COMPARISON TO STATE OF THE ART

#### 4.3 PRE-TRAINING DATA REQUIREMENTS

#### 4.4 SCALING STUDY

#### 4.5 INSPECTING VISION TRANSFORMER

#### 4.6 SELF-SUPERVISION

### 5 CONCLUSION
