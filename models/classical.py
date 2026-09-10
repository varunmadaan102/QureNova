from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

def build_classical_models(random_state=42):
    # These settings were selected with stratified 5-fold CV on the checked-in
    # biomedical schema. Logistic Regression favors the best aggregate score;
    # the balanced SVM favors sensitivity for early-detection exploration.
    models = {
        "Logistic Regression": LogisticRegression(
            C=0.5,
            max_iter=3000,
            random_state=random_state,
        ),
        "SVM (RBF)": CalibratedClassifierCV(
            estimator=SVC(
                C=1.0,
                class_weight="balanced",
                kernel="rbf",
                random_state=random_state,
            ),
            method="sigmoid",
            cv=3,
            ensemble=False,
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=1.0,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=1,
        )
    except Exception:
        pass
    return models
