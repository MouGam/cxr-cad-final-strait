# 과제 결과 보고서 캡처 목록

아래 캡처를 `assignment_result_report.md`의 각 삽입 위치에 넣으면 보고서 증빙력이 좋아지고 분량도 15쪽 이상으로 안정적으로 나온다.

## 필수 캡처

1. **GitHub 최종 repo 첫 화면**
   - URL: https://github.com/MouGam/cxr-cad-final-strait
   - README 제목, 프로젝트 설명, Quick Start 일부가 보이게 캡처

2. **GitHub Repository Structure**
   - `app/`, `training/`, `analysis/`, `docs/`, `tests/`, `Dockerfile`이 보이게 캡처

3. **Docker Hub 이미지 페이지**
   - URL: https://hub.docker.com/r/mougam/cxr-cad-final
   - 이미지 이름과 tag 영역이 보이게 캡처

4. **Docker 실행 터미널**
   - 명령어:
     ```bash
     docker pull mougam/cxr-cad-final:latest
     docker run --rm -p 8000:8000 -p 8501:8501 mougam/cxr-cad-final:latest
     ```
   - FastAPI/Streamlit 실행 로그 또는 container running 상태 캡처

5. **Streamlit 대시보드 업로드 화면**
   - `http://localhost:8501`
   - 이미지 업로드 UI가 보이게 캡처

6. **Streamlit 예측 결과 화면**
   - 14개 질환 확률 막대그래프
   - Youden/Screening/Confirmatory threshold marker
   - 탐지 결과 요약이 보이게 캡처

7. **Grad-CAM overlay 화면**
   - 원본 X-ray와 Grad-CAM overlay가 같이 보이게 캡처

8. **Swagger UI**
   - `http://localhost:8000/docs`
   - `/health`, `/predict` endpoint가 보이게 캡처

9. **Health API 응답**
   - `http://localhost:8000/health`
   - 모델 로드 상태가 보이게 캡처

10. **Predict API 응답**
    - Swagger 또는 curl/Postman으로 `/predict` 응답 JSON 일부 캡처
    - `predictions`, `thresholds`, `detected`, `inference_time_ms`, `gradcam_time_ms`가 보이면 좋음

11. **README Model Summary 표**
    - AUROC 0.8520, Serving Best Pair 0.8464가 보이게 캡처

12. **Platt Scaling 결과**
    - `docs/report_evaluation.md`의 Platt Scaling 표 또는 `outputs/platt_scaling_ensemble.png`
    - ECE 0.2331 -> 0.0029가 보이게 캡처

13. **Subgroup Analysis 표**
    - 성별/연령/View Position 차이가 10% 미만임을 보여주는 표 캡처

14. **External Validation 표**
    - CheXpert 10,000장, 평균 AUROC 0.8012, NIH 대비 gap이 보이게 캡처

15. **FP/FN Grad-CAM 분석**
    - `docs/report_xai_error.md` 또는 `outputs/gradcam_errors/`에서 FP/FN 예시 캡처

16. **pytest 결과**
    - `pytest -q` 실행 후 `32 passed`가 보이게 터미널 캡처

17. **HuggingFace 모델 저장소**
    - DenseNet/EfficientNet 모델 repo 중 1~2개 캡처

18. **HuggingFace Dataset 저장소**
    - NIH processed dataset 또는 CheXpert test set 페이지 캡처

## 선택 캡처

1. `docs/preprocessing_report.md`의 CLAHE 전후 비교 이미지
2. 수동 crop 비교 이미지
3. 질환별 AUROC/AUPRC 표
4. Docker Desktop 컨테이너 상태
5. GitHub commit history

## 보고서 삽입 팁

- 요약서는 캡처를 넣지 않는 것이 좋다.
- 결과보고서는 각 캡처 아래에 2~3줄 설명을 붙이면 분량과 가독성이 모두 좋아진다.
- 캡처는 한 페이지에 1~2개 정도만 넣고, 글자가 읽히도록 크게 배치한다.
- 멘토 의견은 멘토 회신 후 기업체 멘토 의견 항목에 별도 삽입한다.
