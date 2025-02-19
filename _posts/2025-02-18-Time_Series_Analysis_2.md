---
layout: single
title:  "Time Series Analysis (2): MA, AR, ARMA, ARIMA, SARIMA, SARIMAX, VAR"
---

## 3. Modeling a Moving Average (MA) Process
### 3.1 Define a Moving Average (MA) Process
- Moving Average Model
 - 현재값이 현재와 과거 오차에 선형적으로 비례한다고 정의, 오차는 백색소음과 같이 상호 독립적이며 정규분포를 가정.
 - Moving Average Model은 MA(q)로 표시하고 q는 차수를 의미한다.
   - 모델의 차수 q는 현잿값에 영향을 미치는 과거 오차 항의 개수를 결정한다.
   - q가 클수록 더 많은 과거 오차 항이 현잿값에 영향을 미친다. (적절한 q 설정이 중요)
 - Moving Average Process에서 현잿값은 수열의 평균, 현재 오차 항, 과거 오차 항으로부터 선형적으로 도출한다.
 - 수식: $y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q} $
 - $\mu: 수열의 평균, \epsilon_t: 현재 오차 항, \epsilon_{t-q}: 과거 오차 항$

- Moving Average Process에서 q(차수) 식별 단계
  1. 데이터 수집
  2. 정상성 test
    2-1 : 정상성 X -> 변환 적용 ex) 차분
    2-2 : 정상성 O -> 다음 단계
  3. ACF 도식화 -> 자기상관계수 식별
    3-1 : 지연 0 이후 유의한 계수를 찾을 수 없으면 -> 확률 보행
    3-2 : 지연 q 이후 갑자기 계수들이 유의하지 않는다면 -> MA(q) process

### 3.2 Moving Average (MA) Process example
- widget sales of xyz widget company
  
## 4. Modeling a AutoRegressive(AR) Process

