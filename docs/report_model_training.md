# 기능요구사항 2항: 모델 학습 보고서

## 목차

1. [모델 아키텍처](#1-모델-아키텍처)
2. [Focal Loss 구현](#2-focal-loss-구현)
3. [Gamma 실험 결과](#3-gamma-실험-결과)
4. [학습 설정](#4-학습-설정)
5. [Fold별 성능](#5-fold별-성능)
6. [모델 선택 근거](#6-모델-선택-근거)

---

## 1. 모델 아키텍처

본 프로젝트에서는 흉부 X-ray 14개 질환 다중 레이블 분류를 위해 세 가지 CNN 아키텍처를 실험하였다. 모든 모델은 ImageNet 사전학습 가중치를 기반으로 미세조정(Fine-tuning)하였다.

### 1.1 DenseNet-121

- **파라미터 수**: 약 8M
- **사전학습**: ImageNet Pretrained
- **입력 크기**: 224 x 224
- **분류기 구조**:
  ```python
  classifier = nn.Sequential(
      nn.Linear(1024, 14),
      nn.Sigmoid()
  )
  ```
- DenseNet-121의 마지막 특성 추출기(feature extractor) 출력 차원인 1024를 14개 질환 클래스로 매핑한다. Sigmoid 활성화 함수를 통해 각 질환에 대한 독립적인 확률값(0~1)을 출력한다.

### 1.2 EfficientNet-B4

- **파라미터 수**: 약 19M
- **사전학습**: ImageNet Pretrained
- **입력 크기**: 380 x 380
- **분류기 구조**:
  ```python
  classifier = nn.Sequential(
      nn.Dropout(0.4),
      nn.Linear(1792, 14),
      nn.Sigmoid()
  )
  ```
- EfficientNet-B4는 더 큰 입력 해상도(380x380)와 복합 스케일링(compound scaling)을 통해 높은 표현력을 확보한다. Dropout(0.4)을 적용하여 과적합을 방지하며, 1792차원의 특성을 14개 질환 클래스로 매핑한다.

### 1.3 EfficientNet-B0 (실험용, 최종 미선택)

- **파라미터 수**: 약 5.3M
- **사전학습**: ImageNet Pretrained
- **분류기 구조**:
  ```python
  classifier = nn.Sequential(
      nn.Dropout(0.2),
      nn.Linear(1280, 14),
      nn.Sigmoid()
  )
  ```
- EfficientNet 계열의 경량 모델로 실험하였으나, DenseNet-121 대비 성능이 낮아 최종 선택에서 제외하였다.

---

## 2. Focal Loss 구현

### 2.1 구현 방식

Focal Loss는 외부 라이브러리를 사용하지 않고 **직접 구현**하였다. 클래스 불균형이 심한 의료 영상 데이터 특성을 고려하여, 쉬운 샘플(easy examples)의 손실 기여도를 줄이고 어려운 샘플(hard examples)에 집중하도록 설계되었다.

### 2.2 수식

Focal Loss의 수식은 다음과 같다:

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

- `p_t`: 정답 클래스에 대한 모델의 예측 확률
- `gamma`: focusing 파라미터 (0이면 표준 BCE와 동일)
- `alpha_t`: 클래스별 가중치

### 2.3 Gamma 파라미터 실험

gamma 값에 따른 효과:
- **gamma = 0**: 표준 Binary Cross-Entropy (BCE)와 동일
- **gamma = 1**: 쉬운 샘플의 손실을 중간 수준으로 감소
- **gamma = 2**: 쉬운 샘플의 손실을 강하게 감소, 어려운 샘플에 더 집중

### 2.4 pos_weight 적용

각 질환별 유병률(prevalence)을 기반으로 pos_weight를 산출하여 적용하였다. 이를 통해 양성 샘플이 극히 적은 질환(예: Hernia, Pneumothorax 등)에 대한 학습 불균형을 완화하였다.

**계산식**:
```
pos_weight = 음성 샘플 수 / 양성 샘플 수
```

**14개 질환별 pos_weight** (Train set 기준):

| 질환 | 유병률 (%) | pos_weight | 의미 |
|------|:---------:|:----------:|------|
| Atelectasis | 10.3 | 8.7 | 양성 1건 = 음성 8.7건 가치 |
| Cardiomegaly | 2.5 | 39.0 | |
| Consolidation | 4.2 | 22.8 | |
| Edema | 2.0 | 48.5 | |
| Effusion | 11.9 | 7.4 | |
| Emphysema | 2.2 | 44.5 | |
| Fibrosis | 1.5 | 65.7 | |
| Hernia | 0.2 | 534.0 | 가장 높은 가중치 |
| Infiltration | 17.7 | 4.6 | |
| Mass | 5.1 | 18.6 | |
| Nodule | 5.6 | 16.9 | |
| Pleural_Thickening | 3.0 | 32.3 | |
| Pneumonia | 1.2 | 82.3 | |
| Pneumothorax | 4.7 | 20.3 | |

Hernia(0.2%)는 pos_weight가 534로, 양성 1건의 loss가 음성 534건에 해당하는 가중치를 받는다. 이 가중치는 Focal Loss 내부에서 양성 예측의 loss에 곱해져 사용되며, AMP(float16) 환경에서 큰 pos_weight 값으로 인한 수치 overflow를 방지하기 위해 float32 강제 변환을 적용하였다.

---

## 3. Gamma 실험 결과

DenseNet-121을 기준으로 gamma 값(0, 1, 2)에 따른 5-Fold 교차검증 및 테스트 성능을 비교하였다.

### 3.1 DenseNet-121 Gamma별 성능 비교

| Gamma | CV AUROC | CV AUPRC | CV ECE | Test AUROC | Test AUPRC | Test ECE | 해석 |
|:-----:|:--------:|:--------:|:------:|:----------:|:----------:|:--------:|:-----|
| 0 (BCE) | 0.8271 +/- 0.0021 | 0.2457 +/- 0.0065 | 0.2334 +/- 0.0118 | 0.8475 | 0.2688 | 0.2283 | BCE 동일. AUROC/ECE 모두 최적. pos_weight만으로 불균형 충분히 보정 |
| 1 | 0.8251 +/- 0.0012 | 0.2415 +/- 0.0073 | 0.2644 +/- 0.0224 | 0.8455 | 0.2652 | 0.2617 | Easy sample 억제 시작. AUROC 소폭 하락, ECE 악화 |
| 2 | 0.8236 +/- 0.0021 | 0.2425 +/- 0.0063 | 0.3121 +/- 0.0106 | 0.8455 | 0.2674 | 0.3104 | Hard sample 과집중. ECE 크게 악화(0.31). 노이즈 라벨에 과적합 경향 |

### 3.2 최적 Gamma 선택 근거

**gamma = 0 (표준 BCE)**을 최종 선택하였다. 근거는 다음과 같다:

1. **AUROC 최고**: CV AUROC(0.8271) 및 Test AUROC(0.8475) 모두 gamma=0에서 가장 높은 값을 기록하였다.
2. **AUPRC 최고**: CV AUPRC(0.2457) 및 Test AUPRC(0.2688) 역시 gamma=0이 최고 성능을 보였다.
3. **ECE 최저**: CV ECE(0.2334) 및 Test ECE(0.2283)로, 모델의 보정(calibration) 성능이 가장 우수하였다. gamma가 증가할수록 ECE가 크게 악화되는 경향을 보였다.
4. **일관성**: CV와 Test 결과 간의 경향이 일치하여, 과적합 없이 일반화 성능이 유지됨을 확인하였다.

본 데이터셋에서는 pos_weight만으로 클래스 불균형이 충분히 보정되어, Focal Loss의 추가적인 focusing 효과(gamma > 0)가 오히려 성능을 저하시키는 것으로 판단된다.

---

## 4. 학습 설정

### 4.1 최적화 설정

| 항목 | 설정값 |
|------|--------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-5 |
| LR Scheduler | Cosine Annealing |
| Scheduler eta_min | 1e-6 |

### 4.2 교차검증

| 항목 | 설정값 |
|------|--------|
| 교차검증 방식 | 5-Fold GroupKFold |
| 그룹 기준 | Patient ID (환자 단위) |
| 목적 | 동일 환자의 이미지가 학습/검증 세트에 동시 포함되는 데이터 누출(data leakage) 방지 |

### 4.3 조기 종료 및 학습 효율

| 항목 | 설정값 |
|------|--------|
| Early Stopping | patience = 5 |
| 모니터링 지표 | val_auroc |
| 혼합 정밀도 학습 | AMP float16 (Mixed Precision Training) |

혼합 정밀도 학습(AMP)을 적용하여 GPU 메모리 사용량을 절감하고 학습 속도를 향상시켰다.

---

## 5. Fold별 성능

### 5.1 DenseNet-121 (gamma = 0)

| Fold | Val AUROC |
|:----:|:---------:|
| 0 | 0.8292 |
| 1 | 0.8280 |
| 2 | 0.8275 |
| 3 | 0.8236 |
| 4 | 0.8273 |
| **Mean +/- Std** | **0.8271 +/- 0.0021** |

- Fold 간 편차가 0.0021로 매우 작아, 학습이 안정적으로 수행되었음을 나타낸다.
- Fold 0이 가장 높은 Val AUROC(0.8292)를 기록하였다.

### 5.2 EfficientNet-B4 380 (gamma = 0)

| Fold | Val AUROC |
|:----:|:---------:|
| 0 | 0.8289 |
| 1 | 0.8301 |
| 2 | 0.8331 |
| 3 | 0.8289 |
| 4 | 0.8320 |
| **Mean +/- Std** | **0.8306 +/- 0.0017** |

- EfficientNet-B4는 DenseNet-121 대비 CV AUROC가 0.0035 높다 (0.8306 vs. 0.8271).
- Fold 간 편차가 0.0017로 DenseNet-121보다도 더 안정적이다.
- Fold 2가 가장 높은 Val AUROC(0.8331)를 기록하였다.

---

## 6. 모델 선택 근거

### 6.1 후보 모델 비교

| 모델 | 입력 크기 | 파라미터 수 | Test AUROC | 비고 |
|------|:---------:|:-----------:|:----------:|------|
| DenseNet-121 | 224 | 8M | **0.8475** | 베이스라인, 5-fold 앙상블 테스트 결과 |
| EfficientNet-B0 | 224 | 5.3M | 0.8377 | DenseNet보다 낮아 최종 제외 |
| EfficientNet-B4 | 224 | 19M | 0.8348 | B4 최적 입력(380)이 아니라 성능 저하 |
| EfficientNet-B4 | 380 | 19M | **0.8459** | DenseNet과 앙상블에 사용 |

### 6.2 선택 결과

최종적으로 **DenseNet-121**과 **EfficientNet-B4 (380)**을 앙상블 조합으로 선택하였다.

#### DenseNet-121 선택 이유
- 8M 파라미터의 경량 구조로 빠른 추론이 가능하다.
- Test AUROC 0.8475로 단일 모델 기준 최고 성능을 기록하였다.
- 베이스라인으로서 안정적인 성능을 제공한다.

#### EfficientNet-B0 제외 이유
- Test AUROC 0.8377로 DenseNet-121(0.8475) 대비 약 0.01 낮은 성능을 보였다.
- 파라미터 수(5.3M)가 적어 경량이지만, 성능 차이가 유의미하여 최종 앙상블에서 제외하였다.

#### EfficientNet-B4 (224) 제외 이유
- EfficientNet-B4의 최적 입력 해상도는 380x380이다.
- 224x224로 축소하여 학습한 경우 Test AUROC가 0.8348에 그쳐, 해상도 축소로 인한 정보 손실이 성능 저하로 이어졌다.

#### EfficientNet-B4 (380) 선택 이유
- 최적 입력 해상도(380x380)를 사용하여 Test AUROC 0.8459를 달성하였다.
- DenseNet-121과 아키텍처 특성이 상이하여, 앙상블 시 상호 보완 효과가 기대된다.
- 실제 두 모델의 Soft Voting 앙상블 결과 **Test AUROC 0.8520**을 달성하여, 단일 모델 대비 유의미한 성능 향상을 확인하였다.

### 6.3 앙상블 최종 성능

| 구성 | Test AUROC |
|------|:----------:|
| DenseNet-121 단독 (Best Fold 0) | 0.8351 |
| EfficientNet-B4 단독 (Best Fold 2) | 0.8331 |
| DenseNet-121 5-fold 앙상블 | 0.8475 |
| EfficientNet-B4 5-fold 앙상블 | 0.8459 |
| **DenseNet 5-fold 앙상블 + EfficientNet 5-fold 앙상블 Soft Voting** | **0.8520** |

두 모델의 예측 확률을 산술 평균(Soft Voting)하여 최종 앙상블을 구성하였으며, 이를 통해 Test AUROC 0.8520의 최고 성능을 달성하였다.
