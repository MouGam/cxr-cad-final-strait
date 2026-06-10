# 멘토 의견 요청 메일 초안

## 제목

캡스톤디자인 프로젝트 최종 평가 의견 요청드립니다 - 팀 스트레이트 전민혁

## 본문

안녕하세요 김성호 멘토님.  
지난 5월 용산에서 뵈었던 멘티 전민혁입니다.

다름이 아니라 캡스톤디자인 프로젝트를 마무리하는 과정에서, 기업체 멘토님께서 프로젝트 결과에 대해 3줄 내외의 평가 의견을 작성해주셔야 하는 항목이 있어 연락드립니다. 바쁘시겠지만 아래 산출물을 확인하신 뒤, 프로젝트 완성도나 보완점, 향후 발전 방향에 대해 간단히 의견을 주시면 최종 결과보고서의 “기업체 멘토 의견” 항목에 반영하겠습니다.

평가를 원활하게 하실 수 있도록 프로젝트 산출물을 아래와 같이 정리하여 전달드립니다.

## 프로젝트 개요

본 프로젝트는 NIH ChestX-ray14 데이터셋을 기반으로 14개 흉부 질환을 동시에 예측하는 CXR-CAD(Chest X-Ray Computer-Aided Detection) 시스템입니다. DenseNet/EfficientNet 기반 multi-label classification 모델을 학습하고, Grad-CAM 설명, calibration, operating point 분석, subgroup analysis, CheXpert external validation, FastAPI 추론 API, Streamlit 판독 보조 대시보드, Docker 기반 실행 환경까지 End-to-End로 구성하였습니다.

주요 결과는 다음과 같습니다.

- 2-Model 5-Fold Ensemble AUROC: 0.8520
- 서빙용 Best Pair(DenseNet fold 0 + EfficientNet-B4 fold 3) AUROC: 0.8464
- Per-disease Platt Scaling 적용 후 ECE: 0.2331 -> 0.0029
- CheXpert 10,000장 external validation 평균 AUROC: 0.8012
- Apple M1 Pro CPU Docker 환경 기준 Ensemble+TTA 추론 시간: 약 470ms
- pytest 32개 통과

## 프로젝트 산출물

1. 깃허브 리포지토리  
https://github.com/MouGam/cxr-cad-final-strait

2. Docker Hub 리포지토리  
https://hub.docker.com/r/mougam/cxr-cad-final

아래 명령어로 바로 컨테이너를 실행하실 수 있습니다.

```bash
docker pull mougam/cxr-cad-final:latest
docker run --rm -p 8000:8000 -p 8501:8501 mougam/cxr-cad-final:latest
```

실행 후 접속 주소는 다음과 같습니다.

- Streamlit 대시보드: http://localhost:8501
- FastAPI Swagger UI: http://localhost:8000/docs
- API health check: http://localhost:8000/health

테스트용 예시 이미지는 GitHub repository의 `sample_data/sample_xray.png`를 사용하시면 됩니다.  
직접 다운로드 링크:  
https://raw.githubusercontent.com/MouGam/cxr-cad-final-strait/main/sample_data/sample_xray.png

3. HuggingFace 모델 및 데이터 저장소

- DenseNet-121 모델 가중치: https://huggingface.co/MouGam/nih-chestxray14-densenet121
- EfficientNet-B0 모델 가중치: https://huggingface.co/MouGam/nih-chestxray14-efficientnet-B0
- EfficientNet-B4 모델 가중치: https://huggingface.co/MouGam/nih-chestxray14-efficientnet-B4
- NIH 전처리 완료 데이터셋: https://huggingface.co/datasets/MouGam/nih-processed-dataset
- CheXpert Domain Shift 테스트셋: https://huggingface.co/datasets/MouGam/chexpert-test-set

4. 최종보고서 및 기술 리포트

- 프로젝트 최종 보고서: https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/project_final_report.md
- 전처리 보고서: https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/preprocessing_report.md
- 모델 학습 보고서: https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_model_training.md
- 성능 평가 보고서: https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_evaluation.md
- 서빙/MLOps 보고서: https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_serving.md
- XAI 및 에러 분석 보고서: https://github.com/MouGam/cxr-cad-final-strait/blob/main/docs/report_xai_error.md

멘토님께서 주시면 좋은 의견 방향은 아래와 같습니다. 편하신 방향으로 3줄 정도만 작성해주셔도 괜찮습니다.

- 프로젝트가 의료 영상 AI 과제로서 어느 정도 완성도를 갖추었는지
- 데이터 전처리, 모델 학습, 설명 가능 AI, 서빙/Docker 구성 중 인상적이거나 보완할 부분
- 향후 실제 서비스 또는 연구 과제로 발전시키기 위해 추가로 고려하면 좋을 점

예시 형식:

> 본 프로젝트는 흉부 X-ray multi-label classification을 전처리, 모델 학습, 성능 평가, XAI, Docker 기반 서빙까지 End-to-End로 구현한 점이 인상적입니다.  
> 특히 calibration, external validation, Grad-CAM error analysis를 포함하여 단순 모델 성능 외의 신뢰성 요소를 함께 검토한 점이 좋았습니다.  
> 향후에는 실제 임상 적용을 위해 더 다양한 기관 데이터 검증과 사용자 UI 개선, 운영 로그 기반 모니터링을 보완하면 좋겠습니다.

바쁘신 와중에 검토해주셔서 감사합니다.  
편하신 때에 회신 주시면 최종 결과보고서에 반영하겠습니다.

감사합니다.  
전민혁 드림
