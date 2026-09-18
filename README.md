# Hospital 30-Day Readmission Prediction

L2-regularized logistic regression, implemented **from scratch** (numpy only —
no `sklearn.linear_model`, no `sklearn.metrics`) to predict 30-day hospital
readmission risk from patient records.

## Project structure

```
readmission-risk/
├── data/
│   └── patients.csv              # dataset (see Data section)
├── src/
│   ├── preprocessing.py          # one-hot encoding, standardization, stratified split
│   ├── logistic_regression.py    # LogisticRegressionL2: gradient descent, L2 penalty
│   ├── metrics.py                # confusion matrix, ROC/AUC, PR curve, decision-cost curve
│   └── train.py                  # end-to-end pipeline: load → train → evaluate → plot
├── outputs/
│   └── evaluation_plots.png      # ROC curve, PR curve, cost curve (generated on run)
├── requirements.txt
└── README.md
```

## Setup & run

```bash
pip install -r requirements.txt
python src/train.py
```

## Data

`data/patients.csv` — 5,000 patient records with columns:

| Column | Description |
|---|---|
| `age` | Patient age |
| `gender` | Male / Female |
| `primary_diagnosis` | Heart Disease, COPD, Diabetes, Kidney Disease, Hypertension |
| `num_procedures` | Number of procedures performed during admission |
| `days_in_hospital` | Length of stay |
| `comorbidity_score` | 0–4 comorbidity burden score |
| `discharge_to` | Home, Home Health Care, Rehabilitation Facility, Skilled Nursing Facility |
| `readmitted` | Target: 1 if readmitted within 30 days, else 0 |

Base readmission rate: **18.8%**, consistent with published Medicare readmission
statistics (~15–20%), so the class imbalance itself is realistic.

## Methodology

- **Preprocessing**: categorical variables (`gender`, `primary_diagnosis`,
  `discharge_to`) are one-hot encoded with `drop_first=True`; numeric features
  are standardized (zero mean, unit variance) using a scaler *fit on the
  training set only*, to avoid test-set leakage into the transform.
- **Split**: 80/20 stratified train/test split (implemented from scratch),
  preserving the 18.8% positive rate in both sets.
- **Model**: `LogisticRegressionL2` — batch gradient descent minimizing
  weighted log loss + L2 penalty:

  ```
  J(w) = -(1/n) Σ [y·log(p) + (1-y)·log(1-p)]  +  (λ / 2n) ‖w‖²
  ```

  The bias term is not regularized (standard convention). Class weighting
  (`class_weight`) upweights the minority (readmitted) class during training
  so gradient descent isn't dominated by the ~81% majority class.
- **Evaluation**: ROC curve/AUC and Precision-Recall curve/Average Precision
  are computed from scratch by sweeping every unique predicted score as a
  threshold — no `sklearn.metrics`.

## Results — and an important honest finding

Running the pipeline on `data/patients.csv` produces:

| Metric | Value |
|---|---|
| Train ROC-AUC | ~0.54 |
| **Test ROC-AUC** | **~0.47** |
| Test PR-AUC (Average Precision) | ~0.18 (≈ base rate) |

**A ROC-AUC at or below 0.5 means the model performs no better than random
guessing on this dataset.** This is not a bug in the implementation — it was
verified directly:

```
Correlation with readmitted:
  age                  0.0016
  num_procedures      -0.0027
  days_in_hospital    -0.0098
  comorbidity_score   -0.0013

Readmission rate by category (all within ~2 percentage points of the 18.8% base rate):
  gender:             Female 19.4%, Male 18.2%
  primary_diagnosis:  ranges 17.8%–19.9% across all 5 diagnoses
  discharge_to:       ranges 17.3%–19.4% across all 4 dispositions
```

None of the available features carry any measurable linear (or even simple
marginal) relationship with the readmission outcome in this dataset. The
gradient-descent loss curve confirms this: training loss converges almost
immediately to ~0.691, barely below `ln(2) ≈ 0.693` (the loss of a model that
just predicts the base rate for everyone), and the learned coefficients are
all close to zero.

**This is a legitimate and important result to report, not a failure to hide.**
In a real clinical modeling project, this is exactly the kind of finding that
should stop a team before deploying a model: an AUC ≈ 0.5 model provides *no*
discriminative value, and using it to allocate limited post-discharge
resources (case management calls, home health referrals) would be no better,
and arguably worse for accountability, than allocating them at random.

**Likely explanation:** this dataset appears to be a synthetic/teaching
dataset in which `readmitted` was generated independently of the recorded
features (age, procedure count, LOS, comorbidity score, diagnosis category,
discharge disposition), rather than from a true underlying clinical process.
Real EHR-derived readmission datasets (e.g., studies using the LACE index or
HOSPITAL score) reliably show AUCs in the 0.65–0.75 range using similar or
even sparser feature sets, so this null result is a property of *this specific
file*, not of logistic regression or L2 regularization as a method.

**What this means for next steps**, in priority order:
1. Confirm data provenance — verify this file reflects genuine EHR extraction
   rather than a placeholder/synthetic table with an unconditioned label.
2. If genuine, the feature set is insufficient — real readmission signal
   typically requires prior-utilization history (admissions/ED visits in the
   past 6–12 months), lab values, vital sign trends, and social determinants
   (insurance status, living situation), none of which are present here.
3. Re-run this exact pipeline once richer features are available — the code
   requires no changes beyond `preprocessing.py`'s column lists.

## Clinical cost of false negatives vs. false positives

Even with a non-predictive model, the cost framework below is how threshold
selection *should* be approached once a model does have real signal, and is
implemented in `metrics.decision_curve()`:

- **False negative** (predicted low-risk, patient is actually readmitted):
  the patient misses a discharge phone call, medication reconciliation, or
  home-health referral that might have prevented a preventable readmission.
  This is the more clinically severe error — a missed intervention risks
  genuine patient harm, plus financial penalties under CMS's Hospital
  Readmissions Reduction Program (HRRP).
- **False positive** (predicted high-risk, patient is not actually
  readmitted): the patient receives extra outreach they didn't strictly
  need. Costs are real but bounded — care-team time and opportunity cost
  (alert fatigue diluting attention from truly at-risk patients), not
  patient harm.

Because FN harm typically outweighs FP cost, `train.py` computes a **cost-
weighted decision curve** (`FN_COST = 5.0`, `FP_COST = 1.0` — a 5:1 ratio
representative of, but not derived from, typical clinical cost asymmetry)
and reports the threshold that minimizes total expected cost, alongside the
default 0.5 threshold, so the tradeoff is quantified rather than asserted.
**This cost ratio is a modeling assumption stated explicitly in `train.py`
and should be replaced with real institutional cost data (readmission
penalty cost, outreach staff-hour cost) before any clinical use.**

## Limitations

- No genuine predictive signal was found in the provided feature set (see
  above) — reported results should be read as a methodology demonstration,
  not a validated clinical tool.
- No external validation cohort — even with real signal, models trained on
  one hospital's data typically need re-validation elsewhere.
- No subgroup fairness analysis (e.g., AUC/calibration by race, insurance
  status, age) — required before any real deployment, given documented bias
  risks in readmission models that use utilization as a proxy for need.
- Calibration (not just discrimination) was not separately assessed here;
  if a probability output is used for resource allocation, a stated 20%
  risk should correspond to an observed ~20% event rate.
