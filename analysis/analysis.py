"""
분석 스크립트: Youden's J, Operating Points, TTA 비교, Subgroup Analysis, Calibration

입력: outputs/preds_*.npy, outputs/labels.npy, outputs/test_metadata.csv
출력: outputs/ 아래 JSON + PNG 파일들
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def load_predictions():
    """저장된 predictions + labels 로드"""
    labels = np.load(OUTPUT_DIR / "labels.npy")
    meta = pd.read_csv(OUTPUT_DIR / "test_metadata.csv")

    preds = {}
    for arch_prefix in ["densenet", "effb4"]:
        for fold in range(5):
            for suffix in ["", "_tta"]:
                key = f"{arch_prefix}_fold{fold}{suffix}"
                path = OUTPUT_DIR / f"preds_{key}.npy"
                if path.exists():
                    preds[key] = np.load(path)

    return labels, preds, meta


def compute_auroc_per_disease(labels, probs):
    scores = []
    for i in range(14):
        if len(np.unique(labels[:, i])) < 2:
            scores.append(np.nan)
        else:
            scores.append(roc_auc_score(labels[:, i], probs[:, i]))
    return np.array(scores)


def compute_auprc_per_disease(labels, probs):
    scores = []
    for i in range(14):
        if len(np.unique(labels[:, i])) < 2:
            scores.append(np.nan)
        else:
            scores.append(average_precision_score(labels[:, i], probs[:, i]))
    return np.array(scores)


def compute_ece(y_true, y_prob, n_bins=15):
    probs = y_prob.flatten()
    labels_flat = y_true.flatten()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(probs)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / total) * abs(probs[mask].mean() - labels_flat[mask].mean())
    return float(ece)


def ensemble_5fold(preds, arch_prefix, tta_suffix=""):
    """5-fold soft voting"""
    fold_preds = [preds[f"{arch_prefix}_fold{f}{tta_suffix}"] for f in range(5)]
    return np.mean(fold_preds, axis=0)


def ensemble_2model(preds_dn, preds_eff):
    """DenseNet + EfficientNet soft voting"""
    return (preds_dn + preds_eff) / 2.0


# ─── 1. TTA 비교 ───

def analyze_tta(labels, preds):
    print("\n" + "=" * 60)
    print("1. TTA 전후 비교")
    print("=" * 60)

    results = {}
    for arch, prefix in [("DenseNet-121", "densenet"), ("EfficientNet-B4", "effb4")]:
        ens_no_tta = ensemble_5fold(preds, prefix, "")
        ens_tta = ensemble_5fold(preds, prefix, "_tta")

        auroc_no = float(np.nanmean(compute_auroc_per_disease(labels, ens_no_tta)))
        auroc_tta = float(np.nanmean(compute_auroc_per_disease(labels, ens_tta)))

        results[arch] = {
            "no_tta": auroc_no,
            "tta": auroc_tta,
            "diff": auroc_tta - auroc_no,
        }
        print(f"  {arch}: TTA OFF={auroc_no:.4f} → TTA ON={auroc_tta:.4f} (diff={auroc_tta-auroc_no:+.4f})")

    # Ensemble (DenseNet f0 + B4 f3)
    dn_f0 = preds["densenet_fold0"]
    dn_f0_tta = preds["densenet_fold0_tta"]
    eff_f3 = preds["effb4_fold3"]
    eff_f3_tta = preds["effb4_fold3_tta"]

    ens_no = ensemble_2model(dn_f0, eff_f3)
    ens_tta = ensemble_2model(dn_f0_tta, eff_f3_tta)

    auroc_no = float(np.nanmean(compute_auroc_per_disease(labels, ens_no)))
    auroc_tta = float(np.nanmean(compute_auroc_per_disease(labels, ens_tta)))
    results["Ensemble (f0+f3)"] = {"no_tta": auroc_no, "tta": auroc_tta, "diff": auroc_tta - auroc_no}
    print(f"  Ensemble (f0+f3): TTA OFF={auroc_no:.4f} → TTA ON={auroc_tta:.4f} (diff={auroc_tta-auroc_no:+.4f})")

    with open(OUTPUT_DIR / "tta_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─── 2. Youden's J Threshold ───

def compute_youden_thresholds(labels, probs):
    """14개 질환별 Youden's J 기반 최적 threshold"""
    thresholds = {}
    details = {}
    for i, name in enumerate(DISEASE_LABELS):
        if len(np.unique(labels[:, i])) < 2:
            thresholds[name] = 0.5
            continue
        fpr, tpr, thresh = roc_curve(labels[:, i], probs[:, i])
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        thresholds[name] = float(thresh[best_idx])
        details[name] = {
            "threshold": float(thresh[best_idx]),
            "sensitivity": float(tpr[best_idx]),
            "specificity": float(1 - fpr[best_idx]),
            "youden_j": float(j_scores[best_idx]),
        }
    return thresholds, details


def analyze_youden(labels, preds):
    print("\n" + "=" * 60)
    print("2. Youden's J Threshold")
    print("=" * 60)

    all_results = {}

    configs = [
        ("DenseNet 5-fold", ensemble_5fold(preds, "densenet", "_tta")),
        ("EfficientNet-B4 5-fold", ensemble_5fold(preds, "effb4", "_tta")),
        ("Ensemble (f0+f3)", ensemble_2model(preds["densenet_fold0_tta"], preds["effb4_fold3_tta"])),
    ]

    for name, probs in configs:
        thresholds, details = compute_youden_thresholds(labels, probs)
        all_results[name] = {"thresholds": thresholds, "details": details}
        print(f"\n  [{name}]")
        for d in DISEASE_LABELS:
            det = details[d]
            print(f"    {d:<22} threshold={det['threshold']:.3f}  sens={det['sensitivity']:.3f}  spec={det['specificity']:.3f}  J={det['youden_j']:.3f}")

    with open(OUTPUT_DIR / "youden_thresholds.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


# ─── 3. Operating Points ───

def compute_operating_points(labels, probs):
    """Sens@Spec90, Spec@Sens90"""
    results = {}
    for i, name in enumerate(DISEASE_LABELS):
        if len(np.unique(labels[:, i])) < 2:
            results[name] = {"sens_at_spec90": np.nan, "spec_at_sens90": np.nan}
            continue
        fpr, tpr, thresh = roc_curve(labels[:, i], probs[:, i])
        spec = 1 - fpr

        # Sens@Spec90: 특이도 >= 0.9인 점에서 최대 민감도
        mask_spec90 = spec >= 0.9
        sens_at_spec90 = float(tpr[mask_spec90].max()) if mask_spec90.any() else 0.0

        # Spec@Sens90: 민감도 >= 0.9인 점에서 최대 특이도
        mask_sens90 = tpr >= 0.9
        spec_at_sens90 = float(spec[mask_sens90].max()) if mask_sens90.any() else 0.0

        results[name] = {
            "sens_at_spec90": sens_at_spec90,
            "spec_at_sens90": spec_at_sens90,
        }
    return results


def analyze_operating_points(labels, preds):
    print("\n" + "=" * 60)
    print("3. Operating Points (Sens@Spec90, Spec@Sens90)")
    print("=" * 60)

    all_results = {}
    configs = [
        ("DenseNet 5-fold", ensemble_5fold(preds, "densenet", "_tta")),
        ("EfficientNet-B4 5-fold", ensemble_5fold(preds, "effb4", "_tta")),
        ("Ensemble (f0+f3)", ensemble_2model(preds["densenet_fold0_tta"], preds["effb4_fold3_tta"])),
    ]

    for name, probs in configs:
        ops = compute_operating_points(labels, probs)
        all_results[name] = ops
        print(f"\n  [{name}]")
        print(f"    {'Disease':<22} {'Sens@Spec90':>12} {'Spec@Sens90':>12}")
        print(f"    {'-'*48}")
        for d in DISEASE_LABELS:
            print(f"    {d:<22} {ops[d]['sens_at_spec90']:>12.3f} {ops[d]['spec_at_sens90']:>12.3f}")

    with open(OUTPUT_DIR / "operating_points.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


# ─── 4. Calibration Curve ───

def plot_calibration_curve(labels, probs, title, save_path, n_bins=15):
    """Reliability Diagram 생성"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    probs_flat = probs.flatten()
    labels_flat = labels.flatten()
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fracs = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs_flat >= lo) & (probs_flat < hi)
        if mask.sum() > 0:
            fracs.append(labels_flat[mask].mean())
        else:
            fracs.append(np.nan)

    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.bar(bin_centers, fracs, width=1/n_bins, alpha=0.5, edgecolor="black", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {save_path}")


def analyze_calibration(labels, preds):
    print("\n" + "=" * 60)
    print("4. Calibration Curve + ECE")
    print("=" * 60)

    results = {}
    configs = [
        ("DenseNet 5-fold TTA", ensemble_5fold(preds, "densenet", "_tta")),
        ("EfficientNet-B4 5-fold TTA", ensemble_5fold(preds, "effb4", "_tta")),
        ("Ensemble (f0+f3) TTA", ensemble_2model(preds["densenet_fold0_tta"], preds["effb4_fold3_tta"])),
    ]

    for name, probs in configs:
        ece = compute_ece(labels, probs)
        results[name] = {"ece": ece}
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
        plot_calibration_curve(labels, probs, f"{name}\nECE = {ece:.4f}",
                               OUTPUT_DIR / f"calibration_{safe_name}.png")
        print(f"  {name}: ECE = {ece:.4f}")

    with open(OUTPUT_DIR / "calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─── 5. Subgroup Analysis ───

def analyze_subgroups(labels, preds, meta):
    print("\n" + "=" * 60)
    print("5. Subgroup Analysis")
    print("=" * 60)

    # Ensemble (f0+f3) TTA 사용
    probs = ensemble_2model(preds["densenet_fold0_tta"], preds["effb4_fold3_tta"])
    results = {}

    # Gender
    print("\n  [Gender]")
    gender_results = {}
    for g in ["M", "F"]:
        mask = meta["Patient Gender"] == g
        if mask.sum() == 0:
            continue
        auroc = float(np.nanmean(compute_auroc_per_disease(labels[mask], probs[mask])))
        gender_results[g] = {"count": int(mask.sum()), "mean_auroc": auroc}
        print(f"    {g}: N={mask.sum()}, Mean AUROC={auroc:.4f}")
    results["gender"] = gender_results

    # Age groups
    print("\n  [Age Groups]")
    age_results = {}
    ages = meta["Patient Age"].values
    for group_name, lo, hi in [("0-40", 0, 40), ("40-60", 40, 60), ("60+", 60, 200)]:
        mask = (ages >= lo) & (ages < hi)
        if mask.sum() == 0:
            continue
        auroc = float(np.nanmean(compute_auroc_per_disease(labels[mask], probs[mask])))
        age_results[group_name] = {"count": int(mask.sum()), "mean_auroc": auroc}
        print(f"    {group_name}: N={mask.sum()}, Mean AUROC={auroc:.4f}")
    results["age"] = age_results

    # View Position
    print("\n  [View Position]")
    view_results = {}
    for v in meta["View Position"].unique():
        mask = meta["View Position"] == v
        if mask.sum() < 10:
            continue
        auroc = float(np.nanmean(compute_auroc_per_disease(labels[mask], probs[mask])))
        view_results[v] = {"count": int(mask.sum()), "mean_auroc": auroc}
        print(f"    {v}: N={mask.sum()}, Mean AUROC={auroc:.4f}")
    results["view_position"] = view_results

    # 10% 이상 차이 체크
    print("\n  [10% 이상 차이 체크]")
    for category, group_data in results.items():
        if len(group_data) < 2:
            continue
        aurocs = [v["mean_auroc"] for v in group_data.values()]
        diff = max(aurocs) - min(aurocs)
        if diff >= 0.10:
            print(f"    WARNING: {category} 차이 {diff:.4f} >= 0.10 → 원인 분석 필요")
        else:
            print(f"    OK: {category} 차이 {diff:.4f} < 0.10")

    with open(OUTPUT_DIR / "subgroup_analysis.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


# ─── Main ───

def main():
    labels, preds, meta = load_predictions()
    print(f"[Loaded] labels: {labels.shape}, predictions: {len(preds)} sets, metadata: {len(meta)} rows")

    tta_results = analyze_tta(labels, preds)
    youden_results = analyze_youden(labels, preds)
    op_results = analyze_operating_points(labels, preds)
    cal_results = analyze_calibration(labels, preds)
    sub_results = analyze_subgroups(labels, preds, meta)

    print("\n" + "=" * 60)
    print("[완료] 분석 결과 저장 완료")
    print(f"  저장 위치: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
