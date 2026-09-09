from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

def build_classical_models(random_state=42):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, random_state=random_state),
        "SVM (RBF)": CalibratedClassifierCV(
            estimator=SVC(kernel="rbf", random_state=random_state),
            method="sigmoid",
            cv=3,
            ensemble=False,
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=1,
        )
    except Exception:
        pass
    return models
