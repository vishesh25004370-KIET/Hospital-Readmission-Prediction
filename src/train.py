"""
train.py

End-to-end pipeline:
1. Load data/patients.csv (real dataset: hospital_readmission_dataset.csv)
2. Preprocess (one-hot encode categoricals, standardize numerics)
3. Stratified train/test split
4. Fit from-scratch L2-regularized logistic regression
5. Evaluate: ROC-AUC, PR-AUC, confusion matrix at 0.5 and at a
   cost-optimal threshold, coefficient / odds-ratio table
6. Save ROC curve + PR curve + decision-cost plots to outputs/

Run:
    python src/train.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import load_data, build_feature_matrix, StandardScaler, train_test_split
from logistic_regression import LogisticRegressionL2
from metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision,
    precision_recall_f1,
    decision_curve,
)

DATA_PATH = "data/patients.csv"
OUTPUT_DIR = "outputs"

# Relative clinical costs for the decision-curve analysis.
# A missed readmission (false negative) is treated as materially more
# costly than an unnecessary outreach call (false positive) -- see
# README for the clinical justification. This ratio is a modeling
# assumption, not a fitted parameter, and should be revisited with
# actual hospital cost data before real deployment.
FN_COST = 5.0
FP_COST = 1.0


def main():
    print("=" * 60)
    print("Hospital 30-Day Readmission Prediction")
    print("L2-Regularized Logistic Regression (from scratch)")
    print("=" * 60)

    # 1. Load
    df = load_data(DATA_PATH)
    print(f"\nLoaded {len(df)} patient records from {DATA_PATH}")
    print(f"Base readmission rate: {df['readmitted'].mean():.1%}")

    # 2. Feature matrix
    X, y, feature_names = build_feature_matrix(df)
    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} columns")
    print(f"Features: {feature_names}")

    # 3. Split (stratified, fit scaler on train only to avoid leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"\nTrain: {len(y_train)} rows ({y_train.mean():.1%} readmitted)")
    print(f"Test:  {len(y_test)} rows ({y_test.mean():.1%} readmitted)")

    # 4. Fit model. class_weight counteracts the ~19% positive rate so the
    # gradient isn't dominated by the majority (not-readmitted) class.
    pos_rate = y_train.mean()
    class_weight = {0: 1.0, 1: (1 - pos_rate) / pos_rate}
    print(f"\nClass weights: {class_weight}")

    model = LogisticRegressionL2(
        learning_rate=0.5,
        n_iterations=3000,
        l2_lambda=1.0,
        class_weight=class_weight,
        verbose=True,
    )
    print("\nTraining...")
    model.fit(X_train_scaled, y_train)

    # 5. Evaluate
    train_probs = model.predict_proba(X_train_scaled)
    test_probs = model.predict_proba(X_test_scaled)

    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    test_ap = average_precision(y_test, test_probs)

    print("\n" + "-" * 60)
    print(f"Train ROC-AUC: {train_auc:.4f}")
    print(f"Test  ROC-AUC: {test_auc:.4f}")
    print(f"Test  PR-AUC (Average Precision): {test_ap:.4f}")
    print(f"(baseline PR-AUC at random = base rate = {y_test.mean():.4f})")

    # Metrics at default 0.5 threshold
    preds_05 = (test_probs >= 0.5).astype(int)
    m_05 = precision_recall_f1(y_test, preds_05)
    print("\n--- Threshold = 0.50 ---")
    print(f"  Sensitivity (recall): {m_05['recall_sensitivity']:.3f}")
    print(f"  Specificity:          {m_05['specificity']:.3f}")
    print(f"  Precision:            {m_05['precision']:.3f}")
    print(f"  F1:                   {m_05['f1']:.3f}")
    print(f"  TP={m_05['tp']}  FP={m_05['fp']}  FN={m_05['fn']}  TN={m_05['tn']}")

    # Cost-optimal threshold (FN weighted 5x FP -- see README)
    dc = decision_curve(y_test, test_probs, fn_cost=FN_COST, fp_cost=FP_COST)
    best_t = dc["best_threshold"]
    preds_best = (test_probs >= best_t).astype(int)
    m_best = precision_recall_f1(y_test, preds_best)
    print(f"\n--- Cost-optimal threshold = {best_t:.2f} (FN cost={FN_COST}x FP cost) ---")
    print(f"  Sensitivity (recall): {m_best['recall_sensitivity']:.3f}")
    print(f"  Specificity:          {m_best['specificity']:.3f}")
    print(f"  Precision:            {m_best['precision']:.3f}")
    print(f"  F1:                   {m_best['f1']:.3f}")
    print(f"  TP={m_best['tp']}  FP={m_best['fp']}  FN={m_best['fn']}  TN={m_best['tn']}")
    print(f"  Total cost at 0.50 threshold: {dc['cost_at_0.5']:.1f}")
    print(f"  Total cost at optimal threshold: {dc['best_cost']:.1f}")

    # Coefficients / odds ratios
    print("\n--- Coefficients (standardized features), sorted by impact ---")
    print(f"{'Feature':<35}{'Coef':>10}{'Odds Ratio':>14}")
    for name, coef, odds in model.coef_summary(feature_names):
        print(f"{name:<35}{coef:>10.3f}{odds:>14.3f}")
    print(f"{'(intercept)':<35}{model.bias_:>10.3f}")

    # 6. Plots
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    axes[0].plot(fpr, tpr, label=f"Model (AUC = {test_auc:.3f})", color="#2563eb", linewidth=2)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Random (AUC = 0.5)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve (Test Set)")
    axes[0].legend()

    # PR
    prec, rec, _ = precision_recall_curve(y_test, test_probs)
    axes[1].plot(rec, prec, color="#16a34a", linewidth=2, label=f"Model (AP = {test_ap:.3f})")
    axes[1].axhline(y_test.mean(), linestyle="--", color="gray", label=f"Random (AP = {y_test.mean():.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve (Test Set)")
    axes[1].legend()

    # Decision cost curve
    axes[2].plot(dc["thresholds"], dc["costs"], color="#dc2626", linewidth=2)
    axes[2].axvline(best_t, linestyle="--", color="gray", label=f"Optimal threshold = {best_t:.2f}")
    axes[2].axvline(0.5, linestyle=":", color="black", alpha=0.5, label="Default threshold = 0.50")
    axes[2].set_xlabel("Classification Threshold")
    axes[2].set_ylabel(f"Total Cost (FN x{FN_COST:.0f}, FP x{FP_COST:.0f})")
    axes[2].set_title("Cost-Weighted Decision Curve")
    axes[2].legend()

    plt.tight_layout()
    fig_path = f"{OUTPUT_DIR}/evaluation_plots.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nSaved evaluation plots -> {fig_path}")


if __name__ == "__main__":
    main()
