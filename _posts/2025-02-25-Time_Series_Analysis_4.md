---
layout: single
title:  "Time Series Analysis (4): SARIMA, SARIMAX, ARIMAX, VAR"
---

## 7. Modeling a SARIMA Process
### 7.1 Define a SARIMA Process
- SARIMA(계절적 자기회귀누적이동평균): ARIMA 모델에서 주기적 패턴을 추가로 고려한 모델
  - 표기: $SARIMA(p,d,q)(P,D,Q)_m$
  - $SARIMA(p,d,q)(0,0,0)_m$은 $ARIMA(p,d,q)$와 동일
  - 매개변수 (p,d,q)는 ARIMA의 (p,d,q)와 동일
  - 매개변수 P는 계절적 AR(P) 프로세스의 차수
  - 매개변수 D는 계절적 적분 차수
  - 매개변수 Q는 계절적 MA(Q) 프로세스의 차수
  - 매개변수 m은 빈도(계절적 주기당 관측 횟수)를 의미
    - 매년, 분기, 월, 주 단위로 기록된 데이터는 주기의 길이를 1년으로 간주한다.
      1. 매년 관찰 데이터 -> m=1
      2. 분기별 관찰 데이터 -> m=4
      3. 월별 기록 데이터 -> m=12
      4. 주간 기록 데이터 -> m=52
    - 일별, 일 하위 시간 단계로 기록되는 데이터 빈도 m
      1. 매일 기록 데이터 -> 주: m=7 / 년: m=365
      2. 시간별 기록 데이터 -> 일: m=24 / 주: m=168 / 년: m=8766
      3. 분당 기록 데이터 -> 시간: 60 / 일: m=1440 / 주: m=10080 / 년: m=525960
      4. 초당 기록 데이터 -> 분: 60 / 시간: 3600 / 일: m=86400 / 주: m=604800 / 년: m=31557600

#### 시계열에서 계절적 패턴 식별하기 => 시계열 분해
  - 시계열 분해 방법: statsmodels 라이브러리의 STL 함수를 사용해 분해 및 시각화 가능
    1. 추세 구성요소: 시계열의 장기적 변화, 시간이 지남에 따라 증가 or 감소하는 시계열 관련
    2. 계절적 구성요소: 시계열의 계절적 패턴, 일정 기간 동안 발생하는 반복적인 변동
    3. 잔차 or 노이즈: 추세 or 계절 구성요소로 설명할 수 없는 불규칙성            
  ![photo 207](/assets/img/blog/img207.png)             
  - 첫 번째 도식은 관찰된 데이터를 보여준다.
  - 두 번째 도식은 추세로 시간이 지남에 따라 증가함을 보여준다.
  - 세 번째 도식은 계절적 구성요소로 시간이 지남에 따라 반복되는 패턴을 확인할 수 있다.
  - 네 번째 도식은 추세 또는 계절적 요소로 설명할 수 없는 데이터 변동인 잔차를 보여준다.
  - 만약, 계절적 구성요소에 계절적 패턴이 없다면 도식은 수평선에 가까울 것이다.
  - 계절적 구성요소는 관측 데이터에서 최고점과 최저점을 생성하므로 그 값이 양수 or 음수이다.

#### $SARIMA(p,d,q)(P,D,Q)_m$ 식별 단계
  1. 데이터 수집
  2. 정상성 test
    - 정상성 X -> 변환 적용 ex) 차분
    - 정상성 O -> 다음 단계
  3. 차분 횟수(d) 설정
  4. 계절적 차분을 통한 차수 횟수(D) 설정
  5. p,q,P,Q 조합 만들기 및 모든 $SARIMA(p,d,q)(P,D,Q)_m$ 조합 피팅
  6. AIC가 가장 낮은 모델 선택
  7. 잔차 분석 : 모델의 실제값과 예측값의 차이인 모델의 잔차를 분석
    - Q-Q 도식이 직선을 만족해야 한다.
    - 잔차 간 상관관계가 없어야 한다. -> Ljung-Box test
    - 위의 두 가지 사항을 만족한다면 예측 진행 / 반대로 만족하지 않는다면, 다른 p와 q 조합 시도

### 7.2 ARIMA Process example
- Number of air passengers Data

#### 7.2.1 정상성 test
![photo 208](/assets/img/blog/img208.png)                                    
- 데이터 시각화 결과, 추세가 존재하므로 정상성을 만족하지 않을 가능성이 높고, 뚜렷한 계절적 패턴을 확인할 수 있다.
- ADF test는 statsmodels 라이브러리의 adfuller를 활용한다.   
- ADF test 결과 -> ADF Statistic: 0.815, p-value: 0.992
- ADF 통계량의 절댓값 크기가 작고 p-value가 높기 때문에 귀무가설 채택으로 현재 해당 데이터는 비정상적
- 현재 데이터는 월별 데이터이므로 m=12 설정

#### 7.2.2 변환을 통한 정상성 만족시키기 및 차분 횟수 설정
- 1차 차분을 적용하여 추세 구성요소를 제거하여 안정화시킨다.                          
- 1차 차분 적용 이후 ADF test 결과 -> ADF Statistic: -2.83, p-value: 0.054
- 1차 차분 적용 이후, 수열이 정상성을 만족하지 않으므로 계절적 차분을 적용한다.
- 계절적 차분(m=12) 적용 이후 ADF test 결과 -> ADF Statistic: -17.625, p-value: 0.000
- 계절적 차분 이후, ADF 통계량의 절댓값 크기가 크고 p-value가 낮기 때문에, 귀무가설을 기각하여 정상성을 만족한다.
=> 1차 차분과 1번의 계절적 차분을 적용했으므로 d=1, D=1로 설정한다.

#### 7.2.3 p,q,P,Q 조합 만들기 및 모든 $SARIMA(p,d,q)(P,D,Q)_m$ 조합 피팅
- itertools의 product 함수를 사용하여 가능한 모든 p,q,P,Q 조합 목록을 생성한다.
- SARIMA(p,q) 모델들을 피팅하기 위한 함수를 정의한 뒤, 각 조합에 대한 모델을 데이터에 피팅한다.     
![photo 209](/assets/img/blog/img209.png)                                      
- AIC가 가장 낮은 모델 $SARIMA(2,1,1)(1,1,2)_12$ 선택
- AIC는 모델의 상대적인 품질을 측정하는 척도이므로 절대적인 측정을 위해 잔차 분석을 추가로 진행한다.

#### 7.2.4 잔차 분석
![photo 210](/assets/img/blog/img210.png)                           
- 질적 분석: 잔차, 잔차 분포, Q-Q Plot, ACF
  - 좌상단: 잔차로 추세나 분산 변화를 나타내지 않고 있다.
  - 우상단: 잔차의 분포로 정규분포에 매우 가깝다.
  - 좌하단: Q-Q 도식으로 잔차의 분포가 $y=x$ 직선에 가깝다.
  - 우하단: ACF 도식으로 자기상관계수가 지연 0 이후에 유의한 계수가 나타나지 않는다.                       
  => 정성적 관점으로 보았을 때, 현재 잔차가 백색소음과 유사함을 알 수 있다..
- 양적 분석: Ljung-Box test
  - 모델 잔차의 첫 10개 지연에 대한 Ljung-Box test를 진행
  - 실행하면 반환된 p-value가 모두 0.05보다 크다는 것을 알 수 있다. -> 귀무가설 채택 -> 잔차 상관관계 X          
  => 따라서, 잔차가 백색소음처럼 독립적이고 상관관계가 없다는 것을 알 수 있다.

#### 7.2.5 $SARIMA(2,1,1)(1,1,2)_12$을 사용한 예측 및 결과 해석         
![photo 211](/assets/img/blog/img211.png)
![photo 212](/assets/img/blog/img212.png)                                        
- MAPE(평균절대백분율오차) 결과 2.85로 단순 계절성 모델, 계절성을 고려하지 않은 ARIMA 모델보다 성능이 우수함을 알 수 있다.   

## 8. Modeling a SARIMAX Process
### 8.1 Define a SARIMAX Process
- SARIMAX: 외생 변수(X)의 선형 조합을 SARIMA 모델에 추가한 것
  - SARIMA 모델에 외생 변수 추가를 통해 외생 변수가 시계열의 미래값에 미치는 영향을 모델링할 수 있다.
  - 외생 변수: 예측 변수 or 입력 변수를 설명하는 데 사용되는 변수
  - 내생 변수: 예측하고자 하는 대상 변수를 정의하는 데 사용되는 변수
  - 표기: $SARIMAX(p,d,q)(P,D,Q)_m$
  - 수식: $y_t = SARIMAX(p,d,q)(P,D,Q)_m + \sum_{i=1}^{n} \beta_i X^i_t$
  - SARIMA 모델은 과거 수열값과 오차 항의 선형 조합이므로 선형 모델이다. 이에 다른 외생 변수의 선형 조합을 추가하므로 SARIMAX도 선형 모델이 된다.
  - 선형 모델은 대상을 예측하는 데 중요하지 않은 외생 변수에 대해 0에 가까운 계수를 부여하기 때문에 feature selection을 수행하지 않아도 된다.
  - SARIMAX는 범주형 변수도 외생 변수로 포함할 수 있지만 인코딩이 필요하다.
  - ARIMAX = 계절성은 없지만, 외생 변수가 있는 모델

#### 시계열 예측을 위한 외생 변수 작업
1. 다양한 외생 변수의 조합으로 여러 모델을 훈련하고 어떤 모델이 가장 좋은 예측을 생성하는지 확인한다.
2. 모든 외생 변수를 포함한 뒤 AIC를 사용하여 모델을 선택한다.

#### 회귀 분석에서 p-value를 무시하는 이유
- p-value를 예측 변수와 목표 사이의 상관관계를 판단하는 방법으로 잘못 해석하는 경우가 존재.
- p-value는 계수가 0과 유의하게 다른지를 테스트하는 것 -> 이 값은 예측 변수가 예측에 유용한지 여부를 결정 X       
=> 따라서, p-value를 기준으로 예측 변수를 제거하면 안 된다. AIC가 가장 낮은 모델을 선택하여 단계적으로 수행해야 한다.

#### SARIMAX 유의사항
- 외생 변수를 포함하면 대상에 대한 강력한 예측 변수를 찾을 수도 있으므로 잠재적으로 유용할 수 있다.
- 하지만, 미래의 여러 시간 단계를 예측할 때 외생 변수도 예측해야 하는 상황이 발생한다.
- 예측에는 항상 약간의 오차가 발생하기 때문에 목표 변수를 예측하기 위해 외생 변수를 예측해야 하는 경우, 목표 변수의 예측 오차가 커질 수 있고 이는 미래의 더 많은 시간 단계를 예측할수록 예측 정확도가 빠르게 저하될 수 있다.
- 이를 피하기 위해 미래의 하나의 시간 단계만 예측하고 외생 변수를 관측한 뒤, 미래의 다른 시간을 예측해야 한다.
- 하지만, 외생 변수의 예측이 쉬운 경우 또는 외생 변수를 예측하여 사용해도 괜찮다. 즉, 상황과 사용 가능한 외생 변수에 따라 달라지므로 관련 전문 지식과 엄격한 실험이 중요한 역할을 한다.

#### $SARIMAX(p,d,q)(P,D,Q)_m$ 식별 단계 
  1. 데이터 수집
  2. 정상성 test
    2-1 : 정상성 X -> 변환 적용 ex) 차분
    2-2 : 정상성 O -> 다음 단계
  3. 차분 횟수(d) 설정
  4. 계절적 차분을 통한 차수 횟수(D) 설정
  5. p,q,P,Q 조합 만들기 및 모든 $SARIMAX(p,d,q)(P,D,Q)_m$ 조합 피팅
  6. AIC가 가장 낮은 모델 선택
  7. 잔차 분석 : 모델의 실제값과 예측값의 차이인 모델의 잔차를 분석
    - Q-Q 도식이 직선을 만족해야 한다.
    - 잔차 간 상관관계가 없어야 한다. -> Ljung-Box test
    - 위의 두 가지 사항을 만족한다면 예측 진행 / 반대로 만족하지 않는다면, 다른 p와 q 조합 시도

### 8.2 SARIMAX Process example
- Macro economic Data, USA GDP Prediction

#### 8.2.1 정상성 test
![photo 213](/assets/img/blog/img213.png)                                             
- 데이터 시각화 결과, 양의 추세가 존재하므로 정상성을 만족하지 않을 가능성이 높고, 계절성이 존재하지 않은 가능성이 높다.
- ADF test는 statsmodels 라이브러리의 adfuller를 활용한다.   
- ADF test 결과 -> ADF Statistic: 1.750, p-value: 0.998
- ADF 통계량의 절댓값 크기가 작고 p-value가 높기 때문에 귀무가설 채택으로 현재 해당 데이터는 비정상적
- 현재 데이터는 분기별 수집 데이터이므로 m=4 설정

#### 8.2.2 변환을 통한 정상성 만족시키기 및 차분 횟수 설정
- 1차 차분을 적용하여 추세 구성요소를 제거하여 안정화시킨다.                          
- 1차 차분 적용 이후 ADF test 결과 -> ADF Statistic: -6.306, p-value: 0.000
- 1차 차분 적용 이후, 수열이 정상성 만족하므로 계절적 차분을 추가로 고려하지 않아도 된다.            
=> 따라서 d=1, D=0으로 설정한다.

#### 8.2.3 p,q,P,Q 조합 만들기 및 모든 $SARIMAX(p,1,q)(P,0,Q)_4$ 조합 피팅
- itertools의 product 함수를 사용하여 가능한 모든 p,q,P,Q 조합 목록을 생성한다.
- SARIMAX(p,q) 모델들을 피팅하기 위한 함수를 정의한 뒤, 각 조합에 대한 모델을 데이터에 피팅한다.            
![photo 214](/assets/img/blog/img214.png)                                         
- AIC가 가장 낮은 모델 $SARIMAX(3,1,3)(0,0,0)_4$ 선택 => 계절적 구성요소들이 0이므로 $ARIMAX(3,1,3)$과 같다.
- $SARIMAX(3,1,3)(0,0,0)_4$ 선택한 뒤 summary table을 생성하여 외생 변수와 관련된 계수를 확인해본다.          
![photo 215](/assets/img/blog/img215.png)     
- p-value가 0.712인 realdpi를 제외한 모든 외생 변수의 p-value가 0.05보다 작으므로 realdpi 계수가 0에 가까운 것을 알 수 있다. 또한, 해당 계수의 값이 0.0091이라는 것도 알 수 잇다.
- 하지만, p-value가 대상과 예측 변수의 연관성을 결정짓지는 않으므로 계수는 모델에 유지된다.
- AIC는 모델의 상대적인 품질을 측정하는 척도이므로 절대적인 측정을 위해 잔차 분석을 추가로 진행한다.

#### 8.2.4 잔차 분석
![photo 216](/assets/img/blog/img216.png)                                         
- 질적 분석: 잔차, 잔차 분포, Q-Q Plot, ACF
  - 좌상단: 잔차로 백색소음처럼 시간이 지남에 따라 추세가 없고 일정한 분산을 보인다.
  - 우상단: 잔차의 분포로 정규분포에 매우 가깝다.
  - 좌하단: Q-Q 도식으로 잔차의 분포가 $y=x$ 직선에 가깝다.
  - 우하단: ACF 도식으로 자기상관계수가 지연 0 이후로 유의한 계수가 나타나지 않는다.                               
  => 정성적 관점으로 보았을 때, 현재 잔차가 백색소음과 유사함을 알 수 있다..
- 양적 분석: Ljung-Box test
  - 모델 잔차의 첫 10개 지연에 대한 Ljung-Box test를 진행
  - 실행하면 반환된 p-value가 모두 0.05보다 크다는 것을 알 수 있다. -> 귀무가설 채택 -> 잔차 상관관계 X        
  => 따라서, 잔차가 백색소음처럼 독립적이고 상관관계가 없다는 것을 알 수 있다.

#### 8.2.5 $ARIMAX(3,1,3)$을 사용한 예측 및 결과 해석
- SARIMAX 모델은 최종 예측 시, 예측 오차가 누적될 수 있으므로 바로 다음 시간 단계만 예측하는 것이 적절하다.
- 따라서, 모델을 테스트하기 위해 다음 시간 단계를 여러 번 예측하는 롤링 예측을 활용하여 각 예측의 오차를 평균한다.                    
![photo 217](/assets/img/blog/img217.png)                                           
- MAPE(평균절대백분율오차) 결과 0.7로 단순 마지막 측정값으로 예측하는 모델(0.74)보다 성능이 약간 더 우수함을 알 수 있다.
- 하지만, 비즈니스적 맥락을 고려하면 미국의 실질 GDP를 예측하는 것이므로 0.04% 차이는 수천 달러에 해당한다.
- 따라서 SARIMAX 모델의 사용성을 정당화할 수 있다.


<br>            

참고문헌: TimeSeries Forecasting In Python
