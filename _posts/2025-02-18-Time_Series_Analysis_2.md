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
- Widget sales of xyz widget company Data
#### 3.2.1 정상성 test
![photo 178](/assets/img/blog/img178.png)                
- 데이터 시각화 결과, 추세가 존재하므로 정상성을 만족하지 않을 가능성이 높다.
- ADF test 결과 ->  ADF Statistic: -1.512, p-value: 0.527
- ADF 통계량의 절댓값 크기가 작고 p-value가 높기 때문에 귀무가설 채택으로 현재 해당 데이터는 비정상성

#### 3.2.2 변환을 통한 정상성 만족시키기
- 1차 차분을 적용하여 추세 구성요소를 제거하여 안정화시킨다.               
![photo 179](/assets/img/blog/img179.png)         
- 1차 차분 적용 이후, 시각화 결과 추세 구성요소가 안정화되어 정상성을 만족함을 알 수 있다.
- 1차 차분 적용 이후 ADF test 결과 -> ADF Statistic: -10.577, p-value: 0.000
- ADF 통계량의 절댓값 크기가 크고 p-value가 낮기 때문에, 귀무가설을 기각하여 정상성을 만족한다.

#### 3.2.3 자기상관함수(ACF) 도식화
- statsmodels 라이브러리의 plot_acf를 활용한다.                   
![photo 180](/assets/img/blog/img180.png)               
- lag(지연) 2까지 유의한 계수가 존재하고 그 이후로는 계수가 유의하지 않는다. => q=2
- ACF 도식을 통해 지연 q까지 유의한 자기상관계수를 확인하고 해당 지연 이후에는 몯느 계수가 유의하지 않음을 확인할 수 있다.
- 특정 지연 이후, 모든 계수가 유의하지 않다면 MA(q) Process가 있다고 결론 내릴 수 있다.

#### 3.2.4 MA Model을 이용한 예측
- MA 모델은 한번에 여러 단계를 예측할 수 없다. -> 과거 오차 항에 선형적으로 의존하고 이 항은 데이터 집합에서 관찰되지 않으므로 재귀적으로 추정해야 하기 때문이다.
- 즉, MA 모델은 앞으로의 q 단계까지만 예측 가능하다. 그 이후의 예측에 대해서는 모델이 과거 오차항 없이 평균만으로 예측하여 평귱만 반환다.
- q 단계 이후 단순 평균 예측을 막기 위해 롤링 예측(rolling forecast)을 사용한다.
- 롤링 예측: 여러 단계를 예측할 때까지 반복해서 q번의 시간 단계씩 예측을 진행                  
![photo 181](/assets/img/blog/img181.png)                    
![photo 182](/assets/img/blog/img182.png)                       
- MSE 결과 1.95로 평균 예측 모델, 이전값 기반 예측 모델보다 성능이 좋게 나왔음을 알 수 있다.             

#### 3.2.5 최종 결과 해석 
- 위에서 차분된 데이터 집합으로 예측을 수행했기 때문에 역변환을 통해 원래 규모로 되돌려야 한다.                
![photo 183](/assets/img/blog/img183.png)                            
- 역변환을 통해 원래 규모의 예측값을 얻을 수 있다.
- 현재 예측 곡선은 일반적인 추세는 잘 따르지만, 가장 큰 최저점과 최고점은 예측하지 못한다는 한계가 있다.
- 역변환 이후 MAE 결과 2.32로 데이터 단위 1000 달러임을 고려했을 때 현재 예측값은 평균적으로 2320 달러 정도 벗어났음을 알 수 있다.
- MAE 사용 이유: 예측값과 실젯값 간 절대 차이의 평균을 반환하므로 해석 용이

## 4. Modeling a AutoRegressive(AR) Process



참고문헌: TimeSeries Forecasting In Python
