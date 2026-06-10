# 과제 결과 요약서 초안

> 사용 방법: `과제결과요약서양식.png`의 항목에 맞춰 붙여넣기 위한 1쪽 이내 요약본이다. 양면 인쇄 금지 조건을 고려하여 문단을 길게 늘리지 않았다. 기업체 멘토 의견은 멘토 회신 수령 후 추가한다.

## ■ 과제 내용

본 과제는 NIH ChestX-ray14 데이터셋을 활용하여 흉부 X-ray 이미지 1장에 대해 14개 질환의 존재 확률을 동시에 예측하는 CXR-CAD(Chest X-Ray Computer-Aided Detection) 시스템을 구축한 프로젝트이다. 전처리, 모델 학습, 성능 평가, 설명 가능 AI, FastAPI 추론 API, Streamlit 판독 보조 대시보드, Docker 기반 실행 환경을 하나의 최종 제출물로 통합하였다.

## ■ 과제 목표

14개 흉부 질환 multi-label classification 모델을 학습하여 Mean AUROC 0.80 이상을 달성하고, Grad-CAM 기반 시각적 설명과 operating point 분석을 통해 판독 보조에 필요한 근거를 제공하는 것을 목표로 하였다. 또한 Docker 실행만으로 API 서버와 대시보드를 구동할 수 있는 재현 가능한 MLOps 환경을 구축하는 것을 목표로 하였다.

## ■ 과제 수행 방법

NIH ChestX-ray14 원본 112,120장을 분석하여 CLAHE, 품질 필터링, 수동 처리 이미지 복구, multi-hot encoding, patient-wise split을 수행하였다. DenseNet-121과 EfficientNet 계열 모델을 전이학습하고, Focal Loss gamma 실험, 5-Fold GroupKFold, TTA, 앙상블을 적용하였다. 이후 calibration, subgroup analysis, CheXpert external validation, Grad-CAM FP/FN 분석을 수행하였다. 서빙 단계에서는 ONNX 변환, warm-up, 병렬 추론, Grad-CAM 분리 처리를 통해 CPU Docker 환경에서 500ms 요구사항에 대응하였다.

## ■ 과제 최종 결과물

최종 데이터셋은 111,979장으로 구성되며, 품질 기준 미달 141장을 제거하고 31장을 수동 처리 후 포함하였다. 2-Model 5-Fold Ensemble은 AUROC 0.8520을 달성하였고, 서빙 best pair(DenseNet fold 0 + EfficientNet-B4 fold 3)는 AUROC 0.8464를 기록하였다. Per-disease Platt Scaling 적용 후 Ensemble ECE는 0.2331에서 0.0029로 개선되었다. CheXpert 10,000장 직접 매핑 7개 질환 external validation에서는 평균 AUROC 0.8012를 기록하였다. 최종 repo에는 FastAPI, Streamlit, Dockerfile, Docker Hub 이미지, pytest 32개 테스트, 기술 리포트가 포함되어 있다.

## ■ 기대효과 및 활용분야

본 과제는 의료 영상 AI 개발에서 단순 정확도뿐 아니라 calibration, operating point, subgroup fairness, external validation, XAI/error analysis, Docker 기반 배포 가능성을 함께 검토한 사례이다. 교육용 의료 AI, 판독 보조 시스템 프로토타입, 의료 영상 AI MLOps 실습, multi-label classification 및 설명 가능 AI 학습 자료로 활용할 수 있다. 실제 임상 진단용이 아니라 교육 및 연구 목적의 판독 보조 프로토타입으로 한정한다.

## ■ 기업체 멘토 의견

추후 김성호 멘토님 평가 의견 수령 후 3줄 내외로 추가 예정.
