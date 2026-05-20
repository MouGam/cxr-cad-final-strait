# Training Guide — NIH ChestX-ray14

## 1. 베이스 이미지

| 항목 | 스펙 |
|------|------|
| OS | Ubuntu 22.04 |
| CUDA | 12.1+ |
| cuDNN | 9+ |
| Python | 3.10+ |
| PyTorch | 2.3+ |
| GPU | NVIDIA (VRAM 16GB+ 권장) |

> Docker 사용 시: `pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime`

---

## 1-1. Docker + JupyterLab 환경 (클라우드 VM)

```bash
# 비밀번호 설정
export JUPYTER_PASSWORD="chestxray2026"

# Docker 실행 (백그라운드, GPU, 공유메모리 확장)
sudo docker run -d --runtime=nvidia --gpus all --shm-size 10g \
  --network=host \
  -e JUPYTER_PASSWORD=$JUPYTER_PASSWORD \
  -v $(pwd)/scripts:/workspace/scripts \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  bash -c "
    pip install pandas scikit-learn opencv-python-headless Pillow tqdm 'numpy<2' tensorboard &&
    tensorboard --logdir /workspace/outputs/tensorboard --host 0.0.0.0 --port 6006 &
    jupyter lab --ip=0.0.0.0 --allow-root --port=8888 --no-browser \
      --NotebookApp.password=\$(python3 -c \"from jupyter_server.auth import passwd; print(passwd('\$JUPYTER_PASSWORD'))\")
  "
```

접속:
- JupyterLab: `http://<서버IP>:8888` (비밀번호: `chestxray2026`)
- TensorBoard: `http://<서버IP>:6006`

> `--network=host`로 포트가 호스트에 직접 바인딩됩니다.
> `--shm-size 10g`는 DataLoader num_workers > 0일 때 필요합니다.
> `-d`로 백그라운드 실행 — `sudo docker logs -f <container_id>`로 로그 확인 가능.

SSH 터널 사용 시:
```bash
ssh -L 8888:localhost:8888 -L 6006:localhost:6006 user@서버IP
```
그 후:
- JupyterLab: `http://localhost:8888`
- TensorBoard: `http://localhost:6006`

---

## 2. 환경 설정

### 2-1. 시스템 패키지

```bash
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### 2-2. Python 패키지

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pandas>=2.0 scikit-learn>=1.3 opencv-python-headless>=4.8 Pillow>=10.0 tqdm>=4.65 "numpy<2"
```

### 2-3. 데이터 준비

HuggingFace에서 전처리 완료된 데이터셋 다운로드 + 압축 해제:

```bash
pip install huggingface_hub[hf_transfer]

# 다운로드 (tar.gz 1개 파일, 병렬 전송)
HF_HUB_ENABLE_HF_TRANSFER=1 python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='MouGam/nih-processed-dataset', repo_type='dataset', local_dir='./data_download')
"

# 압축 해제
mkdir -p data
tar xzf data_download/nih-processed-dataset.tar.gz -C data
```

해제 후 구조:
```
data/
├── available/
│   ├── images/       # 111,979장 (224x224x3 PNG)
│   ├── train.csv     # 96,359 rows (fold 0~4 포함)
│   └── test.csv      # 15,620 rows
└── unavailable/
    └── ...
```

### 2-4. 스크립트 복사

로컬에서 아래 파일들을 서버로 복사:

```
training/chestxray_train.py
training/run_training.sh
```

---

## 3. 트레이닝 실행

### 3-1. gamma별 개별 실행

```bash
# gamma=0 (BCE 동일)
python training/chestxray_train.py \
  --train_csv data/available/train.csv \
  --test_csv data/available/test.csv \
  --image_dir data/available/images \
  --output_dir outputs/gamma_0_run \
  --gammas 0 \
  --epochs 50 \
  --batch_size 32 \
  --num_folds 5 \
  --workers 4 \
  --patience 5

# gamma=1
python training/chestxray_train.py \
  --train_csv data/available/train.csv \
  --test_csv data/available/test.csv \
  --image_dir data/available/images \
  --output_dir outputs/gamma_1_run \
  --gammas 1 \
  --epochs 50 \
  --batch_size 32 \
  --num_folds 5 \
  --workers 4 \
  --patience 5

# gamma=2
python training/chestxray_train.py \
  --train_csv data/available/train.csv \
  --test_csv data/available/test.csv \
  --image_dir data/available/images \
  --output_dir outputs/gamma_2_run \
  --gammas 2 \
  --epochs 50 \
  --batch_size 32 \
  --num_folds 5 \
  --workers 4 \
  --patience 5
```

### 3-2. 전체 자동 실행 (권장)

```bash
bash training/run_training.sh
```

### 3-3. 회수할 결과물

트레이닝 완료 후 `outputs/` 디렉토리에 아래 파일들이 생성됨:

```
outputs/
├── gamma_0_run/
│   ├── gamma_0.0/
│   │   ├── fold_0.pth    # 모델 가중치
│   │   ├── fold_1.pth
│   │   ├── fold_2.pth
│   │   ├── fold_3.pth
│   │   └── fold_4.pth
│   └── results.json      # CV + Test 성능 결과
├── gamma_1_run/
│   └── ...
└── gamma_2_run/
    └── ...
```

**outputs/ 폴더 전체를 로컬로 가져오면 됨.**
