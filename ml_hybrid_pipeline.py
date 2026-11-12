"""
File: ml_hybrid_pipeline.py
Project: Smart Health Monitoring System (REVA University - R24EJ806)
Author: Vasanth A N
Description: Implements quality-aware preprocessing, rolling feature engineering,
             z-score standardization, subject-independent ML training/evaluation,
             hybrid anomaly scoring (A_t = w_r*R_t + w_m*M_t + w_q*Q_t),
             multi-state alert engine, and reproducible evaluation metrics.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

EPSILON = 1e-8  # Equation (2) stability constant

VITALS = ("hr", "spo2", "temp", "bp_sys", "bp_dia")
MEDIAN_VITALS = ("hr", "spo2")  # pulse-derived values use a rolling median (Sec. 3.3)
FEATURE_KINDS = ("value", "mean", "std", "diff", "trend")  # 5 vitals x 5 = 25 features

# Signal-quality bitmask; must match QualityFlags in esp32_edge_node.ino
QUAL_OK = 0x00
QUAL_PPG_DETACHED = 0x01
QUAL_HR_IMPLAUSIBLE = 0x02
QUAL_SPO2_IMPLAUSIBLE = 0x04
QUAL_TEMP_IMPLAUSIBLE = 0x08
QUAL_BP_IMPLAUSIBLE = 0x10
QUAL_MOTION_ARTIFACT = 0x20
QUAL_ALL = 0x3F

# Flags that make a particular vital unusable for that sample
INVALIDATING_FLAGS = {
    "hr": QUAL_PPG_DETACHED | QUAL_HR_IMPLAUSIBLE,
    "spo2": QUAL_PPG_DETACHED | QUAL_SPO2_IMPLAUSIBLE,
    "temp": QUAL_TEMP_IMPLAUSIBLE,
    "bp_sys": QUAL_BP_IMPLAUSIBLE,
    "bp_dia": QUAL_BP_IMPLAUSIBLE,
}

# Hardware sanity bounds (same as the firmware); values outside are invalid, not abnormal
PLAUSIBLE_RANGES = {
    "hr": (30.0, 220.0),
    "spo2": (70.0, 100.0),
    "temp": (32.0, 43.0),
    "bp_sys": (60.0, 250.0),
    "bp_dia": (35.0, 150.0),
}

ALERT_NORMAL, ALERT_WATCH, ALERT_CRITICAL = "NORMAL", "WATCH", "CRITICAL"


@dataclass
class ClinicalThresholdConfig:
    """Configurable operating ranges (for monitoring prioritization, not diagnosis)."""
    hr_min: float = 50.0
    hr_max: float = 110.0
    spo2_min: float = 94.0
    temp_min: float = 35.5
    temp_max: float = 38.0
    bp_sys_min: float = 90.0
    bp_sys_max: float = 140.0
    bp_dia_min: float = 60.0
    bp_dia_max: float = 90.0
    # Distance beyond a limit at which the rule score R_t saturates at 1.0
    hr_margin: float = 30.0
    spo2_margin: float = 4.0
    temp_margin: float = 1.5
    bp_sys_margin: float = 40.0
    bp_dia_margin: float = 25.0

    def limits(self) -> Dict[str, Tuple[float, float, float]]:
        return {
            "hr": (self.hr_min, self.hr_max, self.hr_margin),
            "spo2": (self.spo2_min, np.inf, self.spo2_margin),
            "temp": (self.temp_min, self.temp_max, self.temp_margin),
            "bp_sys": (self.bp_sys_min, self.bp_sys_max, self.bp_sys_margin),
            "bp_dia": (self.bp_dia_min, self.bp_dia_max, self.bp_dia_margin),
        }


@dataclass
class HybridWeights:
    """Equation (3) weights selected via validation data: w_r + w_m + w_q = 1.0."""
    w_r: float = 0.45  # Rule-based abnormality weight
    w_m: float = 0.40  # ML anomaly score weight
    w_q: float = 0.15  # Signal-quality penalty weight
    watch_threshold: float = 0.40
    critical_threshold: float = 0.70
    persistence_k: int = 3  # Consecutive abnormal samples required to escalate WATCH

    def __post_init__(self) -> None:
        if not np.isclose(self.w_r + self.w_m + self.w_q, 1.0):
            raise ValueError("w_r + w_m + w_q must equal 1.0")
        if not 0.0 < self.watch_threshold < self.critical_threshold <= 1.0:
            raise ValueError("Require 0 < watch_threshold < critical_threshold <= 1")
        if self.persistence_k < 1:
            raise ValueError("persistence_k must be >= 1")


# --------------------------------------------------------------------------
# Preprocessing and feature extraction
# --------------------------------------------------------------------------
def apply_quality_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Replace vitals that are flagged or physiologically implausible with NaN."""
    out = df.copy()
    flags = _flags(out)
    for v in VITALS:
        lo, hi = PLAUSIBLE_RANGES[v]
        col = pd.to_numeric(out[v], errors="coerce")
        bad = ((flags & INVALIDATING_FLAGS[v]) != 0) | (col < lo) | (col > hi)
        out[v] = col.mask(bad)
    return out


def _flags(df: pd.DataFrame) -> pd.Series:
    if "quality_flags" not in df:
        return pd.Series(0, index=df.index, dtype=int)
    return df["quality_flags"].fillna(0).astype(int)


class VitalSignPreprocessor:
    """Implements rolling median/mean filtering, feature extraction, and z-score scaling."""

    def __init__(self, window_n: int = 5, group_col: str = "subject_id"):
        if window_n < 2:
            raise ValueError("window_n must be >= 2")
        self.window_n = window_n
        self.group_col = group_col
        self.mu_: Optional[pd.Series] = None
        self.sigma_: Optional[pd.Series] = None
        # Centered time index for the least-squares slope over a full window
        self._t = np.arange(window_n) - (window_n - 1) / 2.0
        self._t_den = float((self._t ** 2).sum())

    @property
    def feature_names(self) -> List[str]:
        return [f"{v}_{k}" for v in VITALS for k in FEATURE_KINDS]

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-subject rolling features; windows never cross subject boundaries."""
        df = apply_quality_mask(df).reset_index(drop=True)
        groups = df[self.group_col]
        feats: Dict[str, pd.Series] = {}
        for v in VITALS:
            # Carry the last valid value over invalid samples, inside each subject only
            x = df[v].groupby(groups, sort=False).ffill()
            roll = x.groupby(groups, sort=False).rolling(self.window_n, min_periods=1)
            smooth = roll.median() if v in MEDIAN_VITALS else roll.mean()
            smooth = smooth.reset_index(level=0, drop=True)
            feats[f"{v}_value"] = smooth
            feats[f"{v}_mean"] = roll.mean().reset_index(level=0, drop=True)
            feats[f"{v}_std"] = roll.std(ddof=0).reset_index(level=0, drop=True)
            feats[f"{v}_diff"] = smooth.groupby(groups, sort=False).diff()
            feats[f"{v}_trend"] = (
                x.groupby(groups, sort=False)
                .rolling(self.window_n, min_periods=self.window_n)
                .apply(self._slope, raw=True)
                .reset_index(level=0, drop=True)
            )
        return pd.DataFrame(feats, index=df.index)[self.feature_names]

    def _slope(self, w: np.ndarray) -> float:
        return float(np.dot(w - w.mean(), self._t) / self._t_den)

    def fit(self, df: pd.DataFrame) -> "VitalSignPreprocessor":
        """Learn mu and sigma from TRAINING data only (Chapter 4)."""
        feats = self.extract_features(df)
        self.mu_ = feats.mean()
        self.sigma_ = feats.std(ddof=0).fillna(0.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mu_ is None or self.sigma_ is None:
            raise RuntimeError("VitalSignPreprocessor must be fitted before transform()")
        feats = self.extract_features(df)
        z = (feats - self.mu_) / (self.sigma_ + EPSILON)  # Equation (2)
        return z.fillna(0.0)  # missing feature -> training mean

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# --------------------------------------------------------------------------
# Hybrid score components
# --------------------------------------------------------------------------
def rule_scores(
    df: pd.DataFrame, cfg: ClinicalThresholdConfig
) -> Tuple[pd.Series, pd.DataFrame]:
    """R_t in [0, 1]: worst normalized excursion outside the configured ranges.

    Expects a quality-masked frame; invalid (NaN) values contribute no rule score.
    """
    severities = {}
    for v, (lo, hi, margin) in cfg.limits().items():
        x = df[v]
        excess = np.maximum(lo - x, 0.0) + np.maximum(x - hi, 0.0)
        severities[v] = (excess / margin).clip(0.0, 1.0).fillna(0.0)
    sev = pd.DataFrame(severities, index=df.index)
    return sev.max(axis=1), sev


def quality_score(flags: pd.Series) -> pd.Series:
    """Q_t in [0, 1]: data-quality penalty derived from the firmware bitmask."""
    flags = flags.fillna(0).astype(int)
    implausible_bits = sum(
        ((flags & bit) != 0).astype(float)
        for bit in (QUAL_HR_IMPLAUSIBLE, QUAL_SPO2_IMPLAUSIBLE, QUAL_TEMP_IMPLAUSIBLE, QUAL_BP_IMPLAUSIBLE)
    )
    q = 0.5 * ((flags & QUAL_MOTION_ARTIFACT) != 0) + 0.25 * implausible_bits
    q = q.where((flags & QUAL_PPG_DETACHED) == 0, 1.0)
    return q.clip(0.0, 1.0)


class HybridAnomalyDetector:
    """Combines rule score R_t, ML score M_t and quality penalty Q_t (Equation 3)."""

    MODES = ("supervised_rf", "supervised_gb", "unsupervised_iforest")

    def __init__(
        self,
        mode: str = "supervised_rf",
        weights: Optional[HybridWeights] = None,
        thresholds: Optional[ClinicalThresholdConfig] = None,
        window_n: int = 5,
        group_col: str = "subject_id",
        random_state: int = 42,
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode = mode
        self.weights = weights or HybridWeights()
        self.thresholds = thresholds or ClinicalThresholdConfig()
        self.random_state = random_state
        self.preprocessor = VitalSignPreprocessor(window_n, group_col)
        self.model = None
        self._iforest_range: Optional[Tuple[float, float]] = None

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    @property
    def group_col(self) -> str:
        return self.preprocessor.group_col

    def _build_model(self):
        if self.mode == "supervised_rf":
            return RandomForestClassifier(
                n_estimators=200, min_samples_leaf=2, class_weight="balanced",
                random_state=self.random_state, n_jobs=-1,
            )
        if self.mode == "supervised_gb":
            return GradientBoostingClassifier(random_state=self.random_state)
        return IsolationForest(n_estimators=200, contamination="auto", random_state=self.random_state)

    def fit(self, df: pd.DataFrame, y: Optional[Sequence[int]] = None) -> "HybridAnomalyDetector":
        X = self.preprocessor.fit_transform(df)
        model = self._build_model()
        if self.mode.startswith("supervised"):
            if y is None:
                raise ValueError("Supervised modes require labels y")
            model.fit(X, np.asarray(y))
        else:
            # Isolation Forest learns "normal"; drop labelled anomalies when labels exist
            X_fit = X if y is None else X[np.asarray(y) == 0]
            model.fit(X_fit)
            raw = -model.score_samples(X_fit)
            self._iforest_range = (float(np.percentile(raw, 1)), float(np.percentile(raw, 99)))
        self.model = model
        return self

    def ml_score(self, df: pd.DataFrame) -> np.ndarray:
        """M_t in [0, 1]."""
        if not self.is_fitted:
            raise RuntimeError("HybridAnomalyDetector must be fitted before scoring")
        X = self.preprocessor.transform(df)
        if self.mode.startswith("supervised"):
            return self.model.predict_proba(X)[:, 1]
        lo, hi = self._iforest_range
        raw = -self.model.score_samples(X)
        return np.clip((raw - lo) / (hi - lo + EPSILON), 0.0, 1.0)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns R_t, M_t, Q_t, A_t and rule explanation codes for every row of df.

        If no trained model is available, M_t is 0 and the row carries the code
        ML_MODEL_UNAVAILABLE so the omission is visible in the audit log.
        """
        df = df.reset_index(drop=True)
        masked = apply_quality_mask(df)
        r_t, sev = rule_scores(masked, self.thresholds)
        m_t = pd.Series(self.ml_score(df) if self.is_fitted else 0.0, index=df.index)
        q_t = quality_score(_flags(df))
        w = self.weights
        a_t = (w.w_r * r_t + w.w_m * m_t + w.w_q * q_t).clip(0.0, 1.0)

        codes = []
        for i in df.index:
            row_codes = [f"RULE_{v.upper()}_OUT_OF_RANGE" for v in VITALS if sev.at[i, v] > 0]
            if not self.is_fitted:
                row_codes.append("ML_MODEL_UNAVAILABLE")
            codes.append(row_codes)
        return pd.DataFrame({"R_t": r_t, "M_t": m_t, "Q_t": q_t, "A_t": a_t, "rule_codes": codes})

    def save(self, path: Union[str, Path]) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Union[str, Path]) -> "HybridAnomalyDetector":
        obj = joblib.load(path)
        if not isinstance(obj, HybridAnomalyDetector):
            raise TypeError(f"{path} does not contain a HybridAnomalyDetector")
        return obj


# --------------------------------------------------------------------------
# Alert persistence state machine (Section 3.5)
# --------------------------------------------------------------------------
class AlertStateMachine:
    """NORMAL / WATCH / CRITICAL with k-sample persistence and hysteresis."""

    def __init__(self, weights: Optional[HybridWeights] = None):
        self.w = weights or HybridWeights()
        self.state = ALERT_NORMAL
        self.abnormal_run = 0
        self.normal_run = 0

    def update(
        self, a_t: float, quality_flags: int, rule_codes: Sequence[str] = (), r_t: float = 0.0
    ) -> Tuple[str, List[str]]:
        codes = list(rule_codes)
        if quality_flags & (QUAL_MOTION_ARTIFACT | QUAL_PPG_DETACHED):
            # An unreliable sample may neither escalate nor clear an alert
            codes.append("SENSOR_ARTIFACT_REVIEW")
            return self.state, codes

        # Section 3.5: CRITICAL = configured safety-rule violation OR high-confidence anomaly
        if r_t >= 1.0 or a_t >= self.w.critical_threshold:
            self.abnormal_run += 1
            self.normal_run = 0
            self.state = ALERT_CRITICAL
            codes.append("SAFETY_RULE_VIOLATION" if r_t >= 1.0 else "HYBRID_SCORE_CRITICAL")
        elif a_t >= self.w.watch_threshold:
            self.abnormal_run += 1
            self.normal_run = 0
            if self.abnormal_run >= self.w.persistence_k:
                self.state = ALERT_WATCH
                codes.append("PERSISTENT_ANOMALY")
            else:
                codes.append("TRANSIENT_SPIKE_SUPPRESSED")
        else:
            self.abnormal_run = 0
            self.normal_run += 1
            if self.state != ALERT_NORMAL and self.normal_run >= self.w.persistence_k:
                self.state = ALERT_NORMAL
                codes.append("RECOVERED")
        return self.state, codes


def run_alert_engine(
    scored: pd.DataFrame, flags: pd.Series, groups: pd.Series, weights: Optional[HybridWeights] = None
) -> pd.DataFrame:
    """Runs one AlertStateMachine per subject/device over rows in time order."""
    machines: Dict[str, AlertStateMachine] = {}
    states, codes = [], []
    rows = zip(scored["A_t"], scored["R_t"], flags.fillna(0).astype(int), groups, scored["rule_codes"])
    for a_t, r_t, f, g, rc in rows:
        sm = machines.setdefault(g, AlertStateMachine(weights))
        s, c = sm.update(float(a_t), int(f), rc, float(r_t))
        states.append(s)
        codes.append(c)
    return pd.DataFrame({"alert_state": states, "explanation_codes": codes}, index=scored.index)


# --------------------------------------------------------------------------
# Evaluation metrics (Chapter 6)
# --------------------------------------------------------------------------
def subject_level_split(
    df: pd.DataFrame, group_col: str = "subject_id", test_size: float = 0.25, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """GroupShuffleSplit so no subject appears in both partitions."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))
    overlap = set(df[group_col].iloc[train_idx]) & set(df[group_col].iloc[test_idx])
    if overlap:
        raise AssertionError(f"Subject leakage between partitions: {overlap}")
    return train_idx, test_idx


def compute_sensor_mae(reference: Sequence[float], prototype: Sequence[float]) -> float:
    """Equation (4): MAE = (1/n) * sum |y_i - y_hat_i| over paired, non-missing readings."""
    y = np.asarray(reference, dtype=float)
    y_hat = np.asarray(prototype, dtype=float)
    if y.shape != y_hat.shape:
        raise ValueError("reference and prototype must have the same length")
    ok = ~(np.isnan(y) | np.isnan(y_hat))
    if not ok.any():
        raise ValueError("no paired readings")
    return float(np.mean(np.abs(y[ok] - y_hat[ok])))


def compute_latency_statistics(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """L_i = t_receive - t_acquire; reports mean, median, P95 and max (Chapter 6)."""
    lat = np.asarray(latencies_ms, dtype=float)
    lat = lat[~np.isnan(lat)]
    if lat.size == 0:
        raise ValueError("no latency samples")
    return {
        "n": int(lat.size),
        "mean_ms": float(lat.mean()),
        "median_ms": float(np.median(lat)),
        "p95_ms": float(np.percentile(lat, 95)),
        "max_ms": float(lat.max()),
    }


def compute_data_completeness(received_seq_ids: Sequence[int], expected_count: int) -> float:
    """1 - (missing / expected); duplicate deliveries are counted once."""
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    received = len(set(received_seq_ids))
    return 1.0 - (expected_count - min(received, expected_count)) / expected_count


def compute_classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], y_score: Optional[Sequence[float]] = None
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    return out
