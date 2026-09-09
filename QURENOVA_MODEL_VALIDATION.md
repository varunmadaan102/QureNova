# QureNova Model Validation Record

## Scope

This record summarizes the model analysis and tuning performed on the checked-in
30-feature biomedical demo dataset. It is a research engineering record, not a
clinical validation report.

## Evaluation protocol

- Input: `data/demo/qurenova_demo_biomedical.csv`
- Target: `diagnosis`
- Split: stratified five-fold cross-validation
- Random state: 42
- Preprocessing fitted inside each training fold:
  median imputation → StandardScaler → PCA(4)
- Metrics: accuracy, balanced accuracy, precision, recall/sensitivity,
  specificity, F1, ROC-AUC, runtime, and confusion matrix

## Tuning decisions

### Logistic Regression

The tested regularization candidates favored `C=0.5` under balanced accuracy.
The resulting model is the preferred default because it gives the best aggregate
F1 and specificity while remaining fast and interpretable.

### Calibrated RBF SVM

The tested `C` and class-weight candidates showed that `C=1.0` with balanced
weights gives the strongest sensitivity/balanced-accuracy tradeoff. Sigmoid
calibration is retained for probability-like outputs used by patient analysis.

### XGBoost

The tested shallow-tree grid favored 180 estimators, depth 4, learning rate 0.1,
subsample 0.8, and full feature sampling. It remains a nonlinear comparison
model rather than the default model.

## Results

| Model | Accuracy | Balanced accuracy | F1 | Sensitivity | Specificity | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9666 | 0.9608 | 0.9537 | 0.9384 | 0.9831 | 0.9918 |
| Calibrated balanced SVM | 0.9630 | 0.9632 | 0.9528 | 0.9573 | 0.9691 | 0.9917 |
| XGBoost | 0.9596 | 0.9561 | 0.9455 | 0.9432 | 0.9691 | 0.9873 |

## Selection guidance

- Use Logistic Regression as the standard benchmark/default artifact.
- Use SVM when missing positives is the higher-priority error and the
  sensitivity/specificity tradeoff is explicitly accepted.
- Keep XGBoost as a nonlinear baseline and future tuning candidate.
- Do not select a model using accuracy alone.
- Do not interpret demo metrics as prevalence, risk, diagnosis, or clinical
  utility.

## Limitations

The demo is one tabular profile with one target and no external validation. The
metrics do not include confidence intervals, calibration curves, subgroup
fairness, distribution shift, prospective testing, or decision-curve analysis.
Threshold tuning should be performed only after selecting a target operating
point and validation cohort.

The quantum experiment uses a bounded stratified holdout and a simulator-backed
fidelity kernel. It is not a matched comparison with the classical CV protocol.
