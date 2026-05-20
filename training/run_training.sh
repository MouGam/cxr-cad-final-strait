#!/bin/bash
set -e

# ─────────────────────────────────────────────
# NIH ChestX-ray14 Training — gamma 0, 1, 2
# ─────────────────────────────────────────────

DATA_DIR="${DATA_DIR:-data/available}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_FOLDS="${NUM_FOLDS:-5}"
WORKERS="${WORKERS:-4}"
PATIENCE="${PATIENCE:-5}"

WEBHOOK_URL="${WEBHOOK_URL:-}"

send_webhook() {
    local msg="$1"
    if [ -z "$WEBHOOK_URL" ]; then
        return 0
    fi
    curl -s -X POST -H "Content-type: application/json" \
        --data "{\"text\":\"$msg\"}" \
        "$WEBHOOK_URL" > /dev/null 2>&1 || true
}

COMMON_ARGS="--train_csv ${DATA_DIR}/train.csv \
  --test_csv ${DATA_DIR}/test.csv \
  --image_dir ${DATA_DIR}/images \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --num_folds ${NUM_FOLDS} \
  --workers ${WORKERS} \
  --patience ${PATIENCE}"

send_webhook "🚀 트레이닝 시작 (gamma=0,1,2 / epochs=${EPOCHS} / folds=${NUM_FOLDS})"

for GAMMA in 0 1 2; do
    OUTPUT_DIR="${OUTPUT_BASE}/gamma_${GAMMA}_run"

    send_webhook "▶️ gamma=${GAMMA} 트레이닝 시작"
    START=$(date +%s)

    python training/chestxray_train.py \
        ${COMMON_ARGS} \
        --output_dir "${OUTPUT_DIR}" \
        --gammas ${GAMMA}

    END=$(date +%s)
    ELAPSED=$(( (END - START) / 60 ))

    # results.json에서 성능 추출
    if [ -f "${OUTPUT_DIR}/results.json" ]; then
        TEST_AUROC=$(python -c "import json; r=json.load(open('${OUTPUT_DIR}/results.json')); print(f\"{r['test_results'][str(float(${GAMMA}))]['mean_auroc']:.4f}\")")
        send_webhook "✅ gamma=${GAMMA} 완료 (${ELAPSED}분) | Test AUROC: ${TEST_AUROC}"
    else
        send_webhook "⚠️ gamma=${GAMMA} 완료 (${ELAPSED}분) | results.json 없음"
    fi
done

send_webhook "🏁 전체 트레이닝 완료! outputs/ 폴더를 회수하세요."
echo ""
echo "=== All done. Check outputs/ ==="
