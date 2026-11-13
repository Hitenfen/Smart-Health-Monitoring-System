"""
File: run_experiments_and_plots.py
Project: Smart Health Monitoring System (REVA University - R24EJ806)
Author: Vasanth A N
Description: Generates synthetic multi-subject physiological datasets (24 subjects,
             4,800 samples) with controlled clinical anomalies and sensor motion
             artifacts, benchmarks 6 ML models (5 supervised + Isolation Forest)
             under GroupShuffleSplit, evaluates network store-and-forward recovery,
             prints all tables, saves plots to ./results and the trained hybrid
             model to ./hybrid_model.joblib for the cloud server.

All numbers printed by this script come from SYNTHETIC data. They demonstrate the
pipeline; they are not measurements from hardware or human subjects.
"""
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_hybrid_pipeline import (
    QUAL_MOTION_ARTIFACT,
    QUAL_OK,
    QUAL_PPG_DETACHED,
    ClinicalThresholdConfig,
    HybridAnomalyDetector,
    VitalSignPreprocessor,
    compute_classification_metrics,
    compute_data_completeness,
    run_alert_engine,
    subject_level_split,
)

SEED = 42
RESULTS_DIR = Path("results")
MODEL_OUT = Path("hybrid_model.joblib")


def generate_multi_subject_dataset(n_subjects: int = 24, samples_per_subj: int = 200) -> pd.DataFrame:
    """
    Synthesizes continuous multi-parameter vital-sign streams across n_subjects
    with injected hypoxemia, tachycardia+fever, hypertensive spikes, and motion noise.
    """
    rng = np.random.default_rng(SEED)
    records = []
    for s_id in range(1, n_subjects + 1):
        base_hr = rng.uniform(66.0, 82.0)
        base_spo2 = rng.uniform(96.8, 99.0)
        base_temp = rng.uniform(36.4, 37.0)
        base_sys = rng.uniform(112.0, 126.0)
        base_dia = rng.uniform(72.0, 82.0)
        for t in range(samples_per_subj):
            hr = base_hr + rng.normal(0, 2.2)
            spo2 = base_spo2 + rng.normal(0, 0.5)
            temp = base_temp + rng.normal(0, 0.12)
            bp_sys = base_sys + rng.normal(0, 3.5)
            bp_dia = base_dia + rng.normal(0, 2.5)
            q_flag = QUAL_OK
            label = 0  # 0 = Normal, 1 = True Physiological Abnormality

            if 80 <= t <= 115 and s_id % 3 == 0:
                # Pattern A: Gradual Hypoxemia + Compensatory Tachycardia
                progress = min(1.0, (t - 80) / 15.0)
                spo2 -= 7.0 * progress
                hr += 28.0 * progress
                label = 1
            elif 140 <= t <= 170 and s_id % 4 == 1:
                # Pattern B: Febrile Tachycardia
                progress = min(1.0, (t - 140) / 10.0)
                temp += 1.6 * progress
                hr += 24.0 * progress
                label = 1
            elif 30 <= t <= 55 and s_id % 5 == 2:
                # Pattern C: Hypertensive Episode
                bp_sys += 38.0
                bp_dia += 18.0
                label = 1
            else:
                u = rng.random()
                if u < 0.02:
                    # Isolated motion artifact: HR spike with motion flag, NOT an abnormality
                    hr += rng.uniform(40.0, 70.0)
                    q_flag |= QUAL_MOTION_ARTIFACT
                elif u < 0.025:
                    # Finger off the PPG sensor: garbage optical values, flagged
                    hr = rng.uniform(0.0, 250.0)
                    spo2 = rng.uniform(50.0, 100.0)
                    q_flag |= QUAL_PPG_DETACHED

            records.append({
                "subject_id": f"SUBJ_{s_id:02d}", "t": t,
                "hr": hr, "spo2": min(100.0, spo2), "temp": temp,
                "bp_sys": bp_sys, "bp_dia": bp_dia,
                "quality_flags": q_flag, "label": label,
            })
    return pd.DataFrame.from_records(records)


def build_models() -> Dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree (d=6)": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED),
        # SVC(probability=True) is deprecated since scikit-learn 1.9
        "SVM (RBF)": CalibratedClassifierCV(SVC(kernel="rbf", class_weight="balanced"), ensemble=False),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
    }


def benchmark_classifiers(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    prep = VitalSignPreprocessor(window_n=5)
    X_tr = prep.fit_transform(train)  # mu/sigma learned on training subjects only
    X_te = prep.transform(test)
    y_tr, y_te = train["label"].to_numpy(), test["label"].to_numpy()

    rows = []
    for name, model in build_models().items():
        model.fit(X_tr, y_tr)
        t0 = time.perf_counter()
        prob = model.predict_proba(X_te)[:, 1]
        per_sample_ms = (time.perf_counter() - t0) * 1000 / len(X_te)
        m = compute_classification_metrics(y_te, (prob >= 0.5).astype(int), prob)
        rows.append({"model": name, **m, "infer_ms_per_sample": per_sample_ms})

    # Unsupervised baseline: trained on normal training samples only
    iso = IsolationForest(n_estimators=200, random_state=SEED).fit(X_tr[y_tr == 0])
    t0 = time.perf_counter()
    raw = -iso.score_samples(X_te)
    per_sample_ms = (time.perf_counter() - t0) * 1000 / len(X_te)
    threshold = np.percentile(-iso.score_samples(X_tr[y_tr == 0]), 95)
    m = compute_classification_metrics(y_te, (raw >= threshold).astype(int), raw)
    rows.append({"model": "Isolation Forest (unsup.)", **m, "infer_ms_per_sample": per_sample_ms})
    return pd.DataFrame(rows)


def evaluate_hybrid_alerts(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[HybridAnomalyDetector, pd.DataFrame, Dict[str, float]]:
    detector = HybridAnomalyDetector(mode="supervised_rf").fit(train, train["label"])
    scored = detector.score(test)
    alerts = run_alert_engine(scored, test["quality_flags"].reset_index(drop=True),
                              test["subject_id"].reset_index(drop=True), detector.weights)
    out = pd.concat([test.reset_index(drop=True), scored, alerts], axis=1)

    # Static single-sample threshold alarm (baseline) on the raw values
    cfg = ClinicalThresholdConfig()
    static_alarm = np.zeros(len(out), dtype=bool)
    for v, (lo, hi, _) in cfg.limits().items():
        static_alarm |= (out[v] < lo) | (out[v] > hi)
    hybrid_alarm = out["alert_state"] != "NORMAL"

    normal = out["label"] == 0
    stats = {
        "far_static": float(static_alarm[normal].mean()),
        "far_hybrid": float(hybrid_alarm[normal].mean()),
        **{f"hybrid_{k}": v for k, v in compute_classification_metrics(
            out["label"], hybrid_alarm.astype(int), out["A_t"]).items()},
    }
    return detector, out, stats


def simulate_store_and_forward(
    hours: float = 24.0, interval_s: float = 5.0, outages_min=(8, 8, 8),
    loss_prob: float = 0.002, ring_cap: int = 120, drain_per_cycle: int = 10,
) -> Dict[str, float]:
    """Packet-level simulation of Wi-Fi outages with and without the ring buffer."""
    rng = np.random.default_rng(SEED)
    n = int(hours * 3600 / interval_s)
    link_up = np.ones(n, dtype=bool)
    for minutes in outages_min:
        length = int(minutes * 60 / interval_s)
        start = int(rng.integers(0, n - length))
        link_up[start:start + length] = False
    lost_in_flight = rng.random(n) < loss_prob

    # Without buffer: anything not delivered in its own cycle is gone
    received_plain = [i for i in range(n) if link_up[i] and not lost_in_flight[i]]

    # With buffer: oldest-first drain, oldest dropped on overflow
    buffer, received_buf, dropped = [], set(), 0
    for i in range(n):
        buffer.append(i)
        if len(buffer) > ring_cap:
            buffer.pop(0)
            dropped += 1
        if not link_up[i]:
            continue
        for _ in range(min(drain_per_cycle, len(buffer))):
            if rng.random() < loss_prob:
                break  # retry next cycle
            received_buf.add(buffer.pop(0))
    return {
        "expected": n,
        "completeness_no_buffer": compute_data_completeness(received_plain, n),
        "completeness_ring_buffer": compute_data_completeness(received_buf, n),
        "ring_overflow_drops": dropped,
        "still_buffered_at_end": len(buffer),
    }


def plot_results(bench: pd.DataFrame, hybrid: pd.DataFrame, stats: Dict[str, float]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    sup = bench[~bench["model"].str.startswith("Isolation")]
    metrics = ["precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(sup))
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * 0.2, sup[m], width=0.2, label=m)
    ax.set_xticks(x, sup["model"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Classifier comparison, subject-level split (synthetic data)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig2_classifier_comparison.png", dpi=150)
    plt.close(fig)

    # Trace for the first held-out subject that contains the hypoxemia episode
    pattern_a = hybrid[(hybrid["label"] == 1) & (hybrid["t"].between(80, 115))]
    subj = pattern_a["subject_id"].iloc[0] if len(pattern_a) else hybrid["subject_id"].iloc[0]
    tr = hybrid[hybrid["subject_id"] == subj]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(tr["t"], tr["A_t"], label="A_t hybrid")
    ax.plot(tr["t"], tr["M_t"], "--", label="M_t ML")
    ax.axhline(0.70, color="red", lw=0.8, label="CRITICAL 0.70")
    ax.axhline(0.40, color="orange", lw=0.8, label="WATCH 0.40")
    ax.set_xlabel("Sample index (5 s intervals)")
    ax.set_title(f"Hybrid anomaly score trace: {subj} (synthetic)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig3_hybrid_trace.png", dpi=150)
    plt.close(fig)

    rf = bench[bench["model"] == "Random Forest"].iloc[0]
    cm = np.array([[rf["tn"], rf["fp"]], [rf["fn"], rf["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_title("Random Forest confusion matrix")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig4_confusion_matrix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = generate_multi_subject_dataset()
    train_idx, test_idx = subject_level_split(df, "subject_id", test_size=0.25, random_state=SEED)
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    print(f"Dataset: {len(df)} samples, {df['subject_id'].nunique()} subjects, "
          f"{int(df['label'].sum())} abnormal")
    print(f"Test subjects ({test['subject_id'].nunique()}): {sorted(test['subject_id'].unique())}")

    bench = benchmark_classifiers(train, test)
    pd.set_option("display.width", 140)
    print("\nTable: classifier comparison (held-out subjects, synthetic data)")
    print(bench[["model", "precision", "recall", "specificity", "f1", "roc_auc",
                 "tp", "fp", "tn", "fn", "infer_ms_per_sample"]].round(3).to_string(index=False))

    detector, hybrid, stats = evaluate_hybrid_alerts(train, test)
    print("\nFalse-alert rate on normal samples:")
    print(f"  static single-sample threshold: {stats['far_static']:.2%}")
    print(f"  hybrid + persistence (k=3):     {stats['far_hybrid']:.2%}")

    net = simulate_store_and_forward()
    print("\nStore-and-forward simulation (24 h, 3 x 8-min outages):")
    for k, v in net.items():
        print(f"  {k}: {v:.4%}" if isinstance(v, float) else f"  {k}: {v}")

    plot_results(bench, hybrid, stats)
    detector.save(MODEL_OUT)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(
        {"classifiers": bench.to_dict(orient="records"), "hybrid": stats, "network": net},
        indent=2, default=float))
    print(f"\nSaved plots and metrics.json to {RESULTS_DIR.resolve()}; model to {MODEL_OUT.resolve()}")


if __name__ == "__main__":
    main()
